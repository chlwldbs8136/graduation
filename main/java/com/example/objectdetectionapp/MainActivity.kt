package com.example.objectdetectionapp

import android.Manifest
import android.content.res.AssetFileDescriptor
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.Interpreter
import permissions.dispatcher.NeedsPermission
import permissions.dispatcher.RuntimePermissions
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


@RuntimePermissions
class MainActivity : AppCompatActivity() {
    companion object {
        private const val MODEL_FILE_NAME = "ssd_mobilenet_v1.tflite"
        private const val LABEL_FILE_NAME = "coco_dataset_labels.txt"
    }

    private lateinit var overlaySurfaceView: OverlaySurfaceView
    private lateinit var cameraExecutor: ExecutorService

    // tflite모델을 다루기 위한 wrapper를 포함한 interpreter
    private val interpreter: Interpreter by lazy {
        Interpreter(loadModel())
    }

    // 모델의 correct label list
    private val labels: List<String> by lazy {
        loadLabels()
    }

    // camera의 YUV 이미지를 RGB로 변환하는 converter
    private val yuvToRgbConverter: YuvToRgbConverter by lazy {
        YuvToRgbConverter(this)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        overlaySurfaceView = OverlaySurfaceView(resultView)

        cameraExecutor = Executors.newSingleThreadExecutor()
        // permissionDispatcher setUpCamera() method 불러오기
        setupCameraWithPermissionCheck()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        onRequestPermissionsResult(requestCode, grantResults)
    }

    @NeedsPermission(Manifest.permission.CAMERA)
    fun setupCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // preview youth case
            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(cameraView.surfaceProvider) }

            // 후면 카메라 사용
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // object detector의 use case
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(cameraView.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // 최신 camera preview 영상 내보내기
                .build()
                .also {
                    it.setAnalyzer(
                        cameraExecutor,
                        ObjectDetector(
                            yuvToRgbConverter,
                            interpreter,
                            labels,
                            Size(resultView.width, resultView.height)
                        ) { detectedObjectList ->
                            // 결과 표시
                            overlaySurfaceView.draw(detectedObjectList)
                        }
                    )
                }

            try {
                cameraProvider.unbindAll()

                // bined each use case to camerax
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

            } catch (exc: Exception) {
                Log.e("ERROR: Camera", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // asset 폴더에서 tflite 모델 읽기
    private fun loadModel(fileName: String = MODEL_FILE_NAME): ByteBuffer {
        lateinit var modelBuffer: ByteBuffer
        var file: AssetFileDescriptor? = null
        try {
            file = assets.openFd(fileName)
            val inputStream = FileInputStream(file.fileDescriptor)
            val fileChannel = inputStream.channel
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, file.startOffset, file.declaredLength)
        } catch (e: Exception) {
            Toast.makeText(this, "model 파일 가져오기 오류", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            file?.close()
        }
        return modelBuffer
    }

    // model의 label data를 asset에서 가져오기
    private fun loadLabels(fileName: String = LABEL_FILE_NAME): List<String> {
        var labels = listOf<String>()
        var inputStream: InputStream? = null
        try {
            inputStream = assets.open(fileName)
            val reader = BufferedReader(InputStreamReader(inputStream))
            labels = reader.readLines()
        } catch (e: Exception) {
            Toast.makeText(this, "txt 가져오기 오류", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            inputStream?.close()
        }
        return labels
    }
}
