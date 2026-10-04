// InkBench: the Android counterpart of inklab/netime. Times the ink step and the
// two generality stencils as LiteRT models on every delegate a phone offers, and
// the same step as OpenGL ES 3.1 compute shaders (the GPU baseline).
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.inklab.bench"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.inklab.bench"
        minSdk = 29          // GLES 3.1 compute + NNAPI GATHER (API 29)
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            // Benchmarks must be optimized builds; signed with the debug key
            // so `adb install` works without a keystore.
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // Models are memory-mapped straight out of the APK.
    androidResources { noCompress += "tflite" }
    sourceSets["main"].assets.srcDir("../models")
    // The QNN HTP skel libraries must exist as real files in nativeLibraryDir.
    packaging { jniLibs { useLegacyPackaging = true } }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.50.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.50.0")
}
