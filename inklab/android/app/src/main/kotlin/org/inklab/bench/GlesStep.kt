package org.inklab.bench

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.GLES30
import android.opengl.GLES31
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The GPU baseline: the game's Metal ink kernels (ink_kernels.metal, ink_step ..
 * ink_project) ported line for line to OpenGL ES 3.1 compute shaders, plus the
 * two generality stencils. One [submit] runs K simulation steps and ends in
 * glFinish, so the GPU path is timed the same way as a K-step LiteRT call:
 * wall clock per submission, including the sync.
 *
 * Storage differs from Metal in one place: ES 3.1 image load/store has no r16f
 * format, so the scalar fields (pressure, divergence, stencil fields) are r32f.
 * Velocity and dye are rgba16f, as close to Metal's rg16Float/rgba16Float as ES allows.
 */
class GlesStep(private val workload: String, private val sweeps: Int) : AutoCloseable {
    private val w = 128
    private val h = 256
    private val dt = 1f / 30f
    private val jacobi = 26

    private val display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
    private val context: android.opengl.EGLContext
    private val surface: android.opengl.EGLSurface
    private val programs = HashMap<String, Int>()
    private val tex = HashMap<String, Int>()
    private var splatBuf = 0

    init {
        val ver = IntArray(2)
        check(EGL14.eglInitialize(display, ver, 0, ver, 1))
        val cfgs = arrayOfNulls<EGLConfig>(1); val n = IntArray(1)
        EGL14.eglChooseConfig(display, intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, 0x40 /* EGL_OPENGL_ES3_BIT */,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT, EGL14.EGL_NONE), 0, cfgs, 0, 1, n, 0)
        context = EGL14.eglCreateContext(display, cfgs[0], EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE), 0)
        surface = EGL14.eglCreatePbufferSurface(display, cfgs[0],
            intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0)
        check(EGL14.eglMakeCurrent(display, surface, surface, context)) { "eglMakeCurrent" }

        when (workload) {
            "ink" -> {
                for (name in listOf("vel0", "vel1", "dye0", "dye1")) tex[name] = texture(GLES30.GL_RGBA16F)
                for (name in listOf("pr0", "pr1", "div")) tex[name] = texture(GLES30.GL_R32F)
                for ((name, body) in INK_KERNELS) programs[name] = compute(INK_HEADER + body)
                // Two splats per step, the typical gameplay load (the Core ML and
                // LiteRT models take splat fields as inputs, so this keeps work equal).
                val splats = floatArrayOf(
                    40f, 60f, 20f, -10f, 0.9f, 0.4f, 0.2f, 0f, 3f, 0.9f, 0f, 0f,
                    90f, 180f, 0f, 0f, 1f, 1f, 1f, 0f, 4f, 0.6f, 5f, 0f)
                val b = IntArray(1); GLES30.glGenBuffers(1, b, 0); splatBuf = b[0]
                GLES30.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, splatBuf)
                GLES30.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, splats.size * 4, floats(splats),
                    GLES30.GL_STATIC_DRAW)
            }
            "heat", "grayscott" -> {
                for (name in listOf("f0", "f1")) tex[name] = texture(
                    if (workload == "heat") GLES30.GL_R32F else GLES30.GL_RGBA16F)
                programs["stencil"] = compute(stencilSource(workload))
            }
            else -> error("unknown workload $workload")
        }
        seed()
    }

    /** K simulation steps, then glFinish. Returns nothing; the caller times it. */
    fun submit(k: Int) {
        repeat(k) { if (workload == "ink") inkStep() else stencilStep() }
        GLES30.glFinish()
    }

    private var velA = true; private var dyeA = true; private var prA = true; private var fA = true

    private fun inkStep() {
        val vIn = if (velA) "vel0" else "vel1"; val vOut = if (velA) "vel1" else "vel0"
        val dIn = if (dyeA) "dye0" else "dye1"; val dOut = if (dyeA) "dye1" else "dye0"
        // ink_step: advect with the sampler (linear, clamp to edge), damp, splat, walls
        use("ink_step")
        sample(0, tex[vIn]!!); sample(1, tex[dIn]!!)
        image(2, tex[vOut]!!, GLES30.GL_RGBA16F, GLES31.GL_WRITE_ONLY)
        image(3, tex[dOut]!!, GLES30.GL_RGBA16F, GLES31.GL_WRITE_ONLY)
        GLES30.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 4, splatBuf)
        uniforms("ink_step"); dispatch()
        // ink_vorticity: vOut -> vIn
        use("ink_vorticity")
        image(0, tex[vOut]!!, GLES30.GL_RGBA16F, GLES31.GL_READ_ONLY)
        image(1, tex[vIn]!!, GLES30.GL_RGBA16F, GLES31.GL_WRITE_ONLY)
        uniforms("ink_vorticity"); dispatch()
        // ink_divergence: vIn -> div
        use("ink_divergence")
        image(0, tex[vIn]!!, GLES30.GL_RGBA16F, GLES31.GL_READ_ONLY)
        image(1, tex["div"]!!, GLES30.GL_R32F, GLES31.GL_WRITE_ONLY)
        uniforms("ink_divergence"); dispatch()
        // ink_jacobi x 26, warm-started from the previous step's pressure
        use("ink_jacobi")
        uniforms("ink_jacobi")
        repeat(jacobi) {
            val pIn = if (prA) "pr0" else "pr1"; val pOut = if (prA) "pr1" else "pr0"
            image(0, tex[pIn]!!, GLES30.GL_R32F, GLES31.GL_READ_ONLY)
            image(1, tex["div"]!!, GLES30.GL_R32F, GLES31.GL_READ_ONLY)
            image(2, tex[pOut]!!, GLES30.GL_R32F, GLES31.GL_WRITE_ONLY)
            dispatch(); prA = !prA
        }
        // ink_project: vIn - grad p -> vOut
        use("ink_project")
        image(0, tex[vIn]!!, GLES30.GL_RGBA16F, GLES31.GL_READ_ONLY)
        image(1, tex[if (prA) "pr0" else "pr1"]!!, GLES30.GL_R32F, GLES31.GL_READ_ONLY)
        image(2, tex[vOut]!!, GLES30.GL_RGBA16F, GLES31.GL_WRITE_ONLY)
        uniforms("ink_project"); dispatch()
        velA = !velA; dyeA = !dyeA
    }

    private fun stencilStep() {
        val fmt = if (workload == "heat") GLES30.GL_R32F else GLES30.GL_RGBA16F
        use("stencil")
        uniforms("stencil")
        repeat(sweeps) {
            val a = if (fA) "f0" else "f1"; val b = if (fA) "f1" else "f0"
            image(0, tex[a]!!, fmt, GLES31.GL_READ_ONLY)
            image(1, tex[b]!!, fmt, GLES31.GL_WRITE_ONLY)
            dispatch(); fA = !fA
        }
    }

    // --- GL plumbing -------------------------------------------------------

    private var current = 0
    private fun use(name: String) { current = programs[name]!!; GLES30.glUseProgram(current) }

    private fun uniforms(name: String) {
        GLES30.glUniform2i(GLES30.glGetUniformLocation(current, "uSize"), w, h)
        GLES30.glGetUniformLocation(current, "uDt").let { if (it >= 0) GLES30.glUniform1f(it, dt) }
        GLES30.glGetUniformLocation(current, "uVelDamp").let { if (it >= 0) GLES30.glUniform1f(it, 0.3f) }
        GLES30.glGetUniformLocation(current, "uDyeDamp").let { if (it >= 0) GLES30.glUniform1f(it, 0.03f) }
        GLES30.glGetUniformLocation(current, "uVort").let { if (it >= 0) GLES30.glUniform1f(it, 2.5f) }
        GLES30.glGetUniformLocation(current, "uSplats").let { if (it >= 0) GLES30.glUniform1i(it, 2) }
        GLES30.glGetUniformLocation(current, "uVelIn").let { if (it >= 0) GLES30.glUniform1i(it, 0) }
        GLES30.glGetUniformLocation(current, "uDyeIn").let { if (it >= 0) GLES30.glUniform1i(it, 1) }
    }

    private fun dispatch() {
        GLES31.glDispatchCompute((w + 7) / 8, (h + 7) / 8, 1)
        GLES31.glMemoryBarrier(GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT or GLES31.GL_TEXTURE_FETCH_BARRIER_BIT)
    }

    private fun sample(unit: Int, t: Int) {
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0 + unit); GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t)
    }

    private fun image(unit: Int, t: Int, fmt: Int, access: Int) =
        GLES31.glBindImageTexture(unit, t, 0, false, 0, access, fmt)

    private fun texture(fmt: Int): Int {
        val t = IntArray(1); GLES30.glGenTextures(1, t, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t[0])
        GLES30.glTexStorage2D(GLES30.GL_TEXTURE_2D, 1, fmt, w, h)
        // Linear filtering of rgba16f is core in ES 3.0; r32f is never sampled.
        val filter = if (fmt == GLES30.GL_RGBA16F) GLES30.GL_LINEAR else GLES30.GL_NEAREST
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, filter)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, filter)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        return t[0]
    }

    /** Small random state so no path runs on all-zero (possibly short-circuited) data. */
    private fun seed() {
        val rnd = java.util.Random(1)
        for ((name, t) in tex) {
            val four = !name.startsWith("pr") && name != "div" && !(workload == "heat")
            val n = w * h * (if (four) 4 else 1)
            val data = FloatArray(n) { (rnd.nextFloat() - 0.5f) * 0.2f }
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t)
            GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, w, h,
                if (four) GLES30.GL_RGBA else GLES30.GL_RED, GLES30.GL_FLOAT, floats(data))
        }
    }

    private fun compute(src: String): Int {
        val s = GLES30.glCreateShader(GLES31.GL_COMPUTE_SHADER)
        GLES30.glShaderSource(s, src); GLES30.glCompileShader(s)
        val ok = IntArray(1); GLES30.glGetShaderiv(s, GLES30.GL_COMPILE_STATUS, ok, 0)
        check(ok[0] != 0) { "compile: " + GLES30.glGetShaderInfoLog(s) }
        val p = GLES30.glCreateProgram(); GLES30.glAttachShader(p, s); GLES30.glLinkProgram(p)
        GLES30.glGetProgramiv(p, GLES30.GL_LINK_STATUS, ok, 0)
        check(ok[0] != 0) { "link: " + GLES30.glGetProgramInfoLog(p) }
        return p
    }

    private fun floats(a: FloatArray) = ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder())
        .asFloatBuffer().put(a).also { it.position(0) }

    override fun close() {
        EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        EGL14.eglDestroySurface(display, surface); EGL14.eglDestroyContext(display, context)
    }

    private fun stencilSource(name: String): String {
        val fmt = if (name == "heat") "r32f" else "rgba16f"
        val update = if (name == "heat")
            "vec4 x = c + 0.2 * lap;"
        else """
            float u = c.x, v = c.y, uvv = u * v * v;
            vec4 x = vec4(u + 0.16 * lap.x - uvv + 0.035 * (1.0 - u),
                          v + 0.08 * lap.y + uvv - (0.035 + 0.065) * v, 0.0, 0.0);"""
        return """#version 310 es
            layout(local_size_x = 8, local_size_y = 8) in;
            layout($fmt, binding = 0) readonly uniform highp image2D fIn;
            layout($fmt, binding = 1) writeonly uniform highp image2D fOut;
            uniform ivec2 uSize;
            void main() {
                ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                if (p.x >= uSize.x || p.y >= uSize.y) return;
                vec4 c = imageLoad(fIn, p);
                vec4 lap = imageLoad(fIn, ivec2(max(p.x - 1, 0), p.y)) + imageLoad(fIn, ivec2(min(p.x + 1, uSize.x - 1), p.y))
                         + imageLoad(fIn, ivec2(p.x, max(p.y - 1, 0))) + imageLoad(fIn, ivec2(p.x, min(p.y + 1, uSize.y - 1)))
                         - 4.0 * c;
                $update
                imageStore(fOut, p, x);
            }"""
    }

    companion object {
        private const val INK_HEADER = """#version 310 es
            layout(local_size_x = 8, local_size_y = 8) in;
            uniform ivec2 uSize;
            ivec2 cl(ivec2 p) { return clamp(p, ivec2(0), uSize - 1); }
        """

        // Line-for-line ports of ink_kernels.metal's ink_* Metal kernels.
        private val INK_KERNELS = mapOf(
            "ink_step" to """
                uniform highp sampler2D uVelIn;
                uniform highp sampler2D uDyeIn;
                layout(rgba16f, binding = 2) writeonly uniform highp image2D velOut;
                layout(rgba16f, binding = 3) writeonly uniform highp image2D dyeOut;
                struct Splat { vec4 posVel; vec4 color; vec4 params; };
                layout(std430, binding = 4) readonly buffer Splats { Splat splats[]; };
                uniform float uDt; uniform float uVelDamp; uniform float uDyeDamp; uniform int uSplats;
                void main() {
                    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
                    if (gid.x >= uSize.x || gid.y >= uSize.y) return;
                    vec2 sim = vec2(uSize);
                    vec2 p = vec2(gid) + 0.5;
                    vec2 v = texture(uVelIn, p / sim).xy;
                    vec2 back = (p - v * uDt) / sim;
                    vec2 nv = texture(uVelIn, back).xy * exp(-uDt * uVelDamp);
                    vec3 nd = texture(uDyeIn, back).rgb * exp(-uDt * uDyeDamp);
                    for (int i = 0; i < uSplats; i++) {
                        Splat s = splats[i];
                        vec2 d = p - s.posVel.xy;
                        float r = max(s.params.x, 0.5);
                        float g = exp(-dot(d, d) / (r * r));
                        nv += s.posVel.zw * g;
                        float dl = length(d);
                        if (dl > 0.001) nv += (d / dl) * (s.params.z * g);
                        nd += s.color.rgb * (s.params.y * g);
                    }
                    if (gid.x == 0 || gid.y == 0 || gid.x == uSize.x - 1 || gid.y == uSize.y - 1) nv = vec2(0.0);
                    imageStore(velOut, gid, vec4(nv, 0.0, 0.0));
                    imageStore(dyeOut, gid, vec4(min(nd, vec3(1.6)), 1.0));
                }""",
            "ink_vorticity" to """
                layout(rgba16f, binding = 0) readonly uniform highp image2D velIn;
                layout(rgba16f, binding = 1) writeonly uniform highp image2D velOut;
                uniform float uDt; uniform float uVort;
                float curl(ivec2 p) {
                    return 0.5 * ((imageLoad(velIn, cl(p + ivec2(1, 0))).y - imageLoad(velIn, cl(p - ivec2(1, 0))).y)
                                - (imageLoad(velIn, cl(p + ivec2(0, 1))).x - imageLoad(velIn, cl(p - ivec2(0, 1))).x));
                }
                void main() {
                    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                    if (p.x >= uSize.x || p.y >= uSize.y) return;
                    float cC = curl(p);
                    float cL = curl(cl(p - ivec2(1, 0))), cR = curl(cl(p + ivec2(1, 0)));
                    float cU = curl(cl(p - ivec2(0, 1))), cD = curl(cl(p + ivec2(0, 1)));
                    vec2 grad = 0.5 * vec2(abs(cR) - abs(cL), abs(cD) - abs(cU));
                    vec2 nv = imageLoad(velIn, p).xy;
                    float len = length(grad);
                    if (len > 1e-5) { vec2 n = grad / len; nv += uVort * cC * vec2(n.y, -n.x) * uDt; }
                    imageStore(velOut, p, vec4(nv, 0.0, 0.0));
                }""",
            "ink_divergence" to """
                layout(rgba16f, binding = 0) readonly uniform highp image2D vel;
                layout(r32f, binding = 1) writeonly uniform highp image2D div;
                void main() {
                    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                    if (p.x >= uSize.x || p.y >= uSize.y) return;
                    float L = imageLoad(vel, cl(p - ivec2(1, 0))).x, R = imageLoad(vel, cl(p + ivec2(1, 0))).x;
                    float U = imageLoad(vel, cl(p - ivec2(0, 1))).y, D = imageLoad(vel, cl(p + ivec2(0, 1))).y;
                    imageStore(div, p, vec4(0.5 * ((R - L) + (D - U)), 0.0, 0.0, 0.0));
                }""",
            "ink_jacobi" to """
                layout(r32f, binding = 0) readonly uniform highp image2D prIn;
                layout(r32f, binding = 1) readonly uniform highp image2D div;
                layout(r32f, binding = 2) writeonly uniform highp image2D prOut;
                void main() {
                    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                    if (p.x >= uSize.x || p.y >= uSize.y) return;
                    float L = imageLoad(prIn, cl(p - ivec2(1, 0))).x, R = imageLoad(prIn, cl(p + ivec2(1, 0))).x;
                    float U = imageLoad(prIn, cl(p - ivec2(0, 1))).x, D = imageLoad(prIn, cl(p + ivec2(0, 1))).x;
                    imageStore(prOut, p, vec4((L + R + U + D - imageLoad(div, p).x) * 0.25, 0.0, 0.0, 0.0));
                }""",
            "ink_project" to """
                layout(rgba16f, binding = 0) readonly uniform highp image2D velIn;
                layout(r32f, binding = 1) readonly uniform highp image2D pr;
                layout(rgba16f, binding = 2) writeonly uniform highp image2D velOut;
                void main() {
                    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
                    if (p.x >= uSize.x || p.y >= uSize.y) return;
                    float L = imageLoad(pr, cl(p - ivec2(1, 0))).x, R = imageLoad(pr, cl(p + ivec2(1, 0))).x;
                    float U = imageLoad(pr, cl(p - ivec2(0, 1))).x, D = imageLoad(pr, cl(p + ivec2(0, 1))).x;
                    vec2 nv = imageLoad(velIn, p).xy - 0.5 * vec2(R - L, D - U);
                    if (p.x == 0 || p.y == 0 || p.x == uSize.x - 1 || p.y == uSize.y - 1) nv = vec2(0.0);
                    imageStore(velOut, p, vec4(nv, 0.0, 0.0));
                }""",
        )
    }
}
