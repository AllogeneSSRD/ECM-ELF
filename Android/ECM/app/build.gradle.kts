import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localPropertiesFile.inputStream().use { localProperties.load(it) }
}

/** CMake-friendly path (forward slashes). */
fun String.toCmakePath(): String = replace('\\', '/')

val ecmAndroidGmpRoot: String? = sequenceOf(
    localProperties.getProperty("ecm.android.gmp.root"),
    project.findProperty("ecm.android.gmp.root") as String?,
).mapNotNull { it?.trim()?.takeIf(String::isNotEmpty) }
    .firstOrNull()
    ?.toCmakePath()

android {
    namespace = "com.example.ecm"
    compileSdk {
        version = release(37) {
//            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.example.ecm"
        minSdk = 31
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                if (ecmAndroidGmpRoot != null) {
                    arguments += listOf("-DECM_ANDROID_GMP_ROOT=$ecmAndroidGmpRoot")
                }
            }
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    packaging {
        jniLibs {
            // Never ship vendor-pulled OpenCL (wrong ELF page alignment on 16 KB devices).
            excludes += setOf(
                "**/libOpenCL.so",
                "**/libcutils.so",
                "**/libvndksupport.so",
            )
        }
    }
}

val mpaRoot = rootProject.projectDir.parentFile.parentFile

// Production Stage1 resources have one source owner under src/ecm. Android keeps
// its existing logical asset root; standalone benchmark inputs remain tools data.
tasks.register<Copy>("syncEcmStage1Kernels") {
    from(mpaRoot.resolve("src/ecm/opencl/resources"))
    from(mpaRoot.resolve("kernels/opencl")) {
        include("bench/**")
    }
    into(layout.projectDirectory.dir("src/main/assets/kernels/opencl"))
}

tasks.named("preBuild") {
    dependsOn("syncEcmStage1Kernels")
}

if (ecmAndroidGmpRoot != null) {
    logger.lifecycle("ECM: linking GMP from ecm.android.gmp.root=$ecmAndroidGmpRoot")
} else {
    logger.lifecycle(
        "ECM: ecm.android.gmp.root not set — factorization disabled; see docs/DEV_GMP_SETUP.md",
    )
}

dependencies {
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}
