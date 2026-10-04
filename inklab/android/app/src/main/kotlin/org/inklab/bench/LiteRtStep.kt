package org.inklab.bench

import android.content.Context
import com.qualcomm.qti.QnnDelegate
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * One K-step LiteRT model on one backend. [submit] is one interpreter call,
 * i.e. K simulation steps, with inputs and outputs preallocated (as in netime).
 *
 * Backends:
 *   litert-cpu  XNNPACK, 4 threads
 *   litert-gpu  the LiteRT GPU delegate, fp16 allowed (the GPU-via-ML-framework path)
 *   nnapi       NNAPI with its CPU fallback disabled, fp16 allowed; on MediaTek and
 *               Samsung this is the route to the vendor NPU driver
 *   nnapi-fp32  the same with fp16 relaxation off (tests whether the ink graph's
 *               "gather index out of bounds" comes from fp16 index arithmetic)
 *   qnn-htp     Qualcomm's QNN delegate on the Hexagon NPU (HTP), fp16
 *
 * Which nodes a delegate actually claimed is printed by the LiteRT runtime itself
 * ("Replacing N out of M node(s) with delegate ..."); run_android.sh keeps that log.
 */
class LiteRtStep(ctx: Context, asset: String, val backend: String) : AutoCloseable {
    private val delegate: Delegate?
    private val interpreter: Interpreter
    private val inputs: Array<Any>
    private val outputs: Map<Int, Any>

    init {
        val fd = ctx.assets.openFd(asset)
        val model = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        val opts = Interpreter.Options()
        delegate = when (backend) {
            "litert-cpu" -> { opts.setNumThreads(4); opts.setUseXNNPACK(true); null }
            "litert-gpu" -> GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(true)
                .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED))
            "nnapi", "nnapi-fp32" -> NnApiDelegate(NnApiDelegate.Options()
                .setAllowFp16(backend == "nnapi").setUseNnapiCpu(false)
                .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED))
            "qnn-htp" -> QnnDelegate(QnnDelegate.Options().apply {
                setBackendType(QnnDelegate.Options.BackendType.HTP_BACKEND)
                setSkelLibraryDir(ctx.applicationInfo.nativeLibraryDir)
                setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_FP16)
                // Fixed, non-burst mode: the NE has no such knob, so no boost clocks here either.
                setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE)
            })
            else -> error("unknown backend $backend")
        }
        delegate?.let { opts.addDelegate(it) }
        if (backend != "litert-cpu") opts.setUseXNNPACK(false)   // so unclaimed ops are visible, not hidden in XNNPACK
        interpreter = Interpreter(model, opts)
        val rnd = java.util.Random(1)
        inputs = Array(interpreter.inputTensorCount) { i ->
            val n = interpreter.getInputTensor(i).numElements()
            ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).also { b ->
                repeat(n) { b.putFloat((rnd.nextFloat() - 0.5f) * 0.2f) }; b.rewind()
            }
        }
        outputs = (0 until interpreter.outputTensorCount).associateWith { i ->
            ByteBuffer.allocateDirect(interpreter.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
        }
    }

    fun submit() {
        for (b in inputs) (b as ByteBuffer).rewind()
        for (b in outputs.values) (b as ByteBuffer).rewind()
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
    }

    /** The (seeded, fixed) inputs, in tensor order. */
    fun inputFloats(): List<FloatArray> = inputs.map { b ->
        (b as ByteBuffer).duplicate().order(ByteOrder.nativeOrder()).rewind().let { d ->
            FloatArray(d.remaining() / 4).also { (d as ByteBuffer).asFloatBuffer().get(it) }
        }
    }

    /** One line per input and output tensor: index, name, shape. */
    fun describe(): String =
        (0 until interpreter.inputTensorCount).joinToString("") { i -> interpreter.getInputTensor(i).let {
            "in$i ${it.name()} ${it.shape().joinToString("x")}\n" } } +
        (0 until interpreter.outputTensorCount).joinToString("") { i -> interpreter.getOutputTensor(i).let {
            "out$i ${it.name()} ${it.shape().joinToString("x")}\n" } }

    /** The last submission's float outputs, in tensor order (null for non-float tensors). */
    fun outputFloats(): List<FloatArray?> = outputs.entries.sortedBy { it.key }.map { (i, b) ->
        if (interpreter.getOutputTensor(i).dataType() != org.tensorflow.lite.DataType.FLOAT32) null
        else (b as ByteBuffer).duplicate().order(ByteOrder.nativeOrder()).rewind().let { d ->
            FloatArray(d.remaining() / 4).also { (d as ByteBuffer).asFloatBuffer().get(it) }
        }
    }

    override fun close() {
        interpreter.close()
        (delegate as? AutoCloseable)?.close()
    }
}
