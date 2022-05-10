package com.example.objectdetectionapp

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.Image
import android.util.Log
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op

typealias ObjectDetectorCallback = (image: List<DetectionObject>) -> Unit

/**
 * cameraX 물체 감지 영상 해석 use case
 * @param yuvToRgbConverter 카메라 이미지의 버퍼 YUV_420-888에서 RGB 형식으로 변환
 * @param interpreter tflite 모델 조작하는 library
 * @param labels 원본 label list
 * @param resultViewSize 결과 나타내는 surface View 크기
 * @param listener interpreter 결과 list를 callback으로 받기
 */
class ObjectDetector(
    private val yuvToRgbConverter: YuvToRgbConverter,
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val resultViewSize: Size,
    private val listener: ObjectDetectorCallback
) : ImageAnalysis.Analyzer {

    companion object {
        // 모델의 input, output size
        private const val IMG_SIZE_X = 300
        private const val IMG_SIZE_Y = 300
        private const val MAX_DETECTION_NUM = 10

        // 양자화가 완료된 tflite 모델을 사용 -> normalize는 127.5f X
        private const val NORMALIZE_MEAN = 0f
        private const val NORMALIZE_STD = 1f

        // detection 결과 limitation
        private const val SCORE_THRESHOLD = 0.5f
    }

    private var imageRotationDegrees: Int = 0
    private val tfImageProcessor by lazy {
        ImageProcessor.Builder()
            .add(ResizeOp(IMG_SIZE_X, IMG_SIZE_Y, ResizeOp.ResizeMethod.BILINEAR)) // model의 input에 맞게 img resize
            .add(Rot90Op(-imageRotationDegrees / 90)) // ImageProxy 90도 회전 보정
            .add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD)) // normalization
            .build()
    }

    private val tfImageBuffer = TensorImage(DataType.UINT8)

    // detection result bounding box [1:10:4]
    // bounding box [top, left, bottom, right]
    private val outputBoundingBoxes: Array<Array<FloatArray>> = arrayOf(
        Array(MAX_DETECTION_NUM) {
            FloatArray(4)
        }
    )

    // detection 결과의 class label index [1:10]
    private val outputLabels: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )

    // detection 결과의 score [1:10]
    private val outputScores: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )

    // detection object 수(tflite 변환시에 설정되어 있음(일정))
    private val outputDetectionNum: FloatArray = FloatArray(1)

    // detection 결과를 위한 map 정리
    private val outputMap = mapOf(
        0 to outputBoundingBoxes,
        1 to outputLabels,
        2 to outputScores,
        3 to outputDetectionNum
    )

    // cameraX preview의 image를 object detection model에 넣어 추론
    @SuppressLint("UnsafeExperimentalUsageError")
    override fun analyze(image: ImageProxy) {
        if (image.image == null) return
        imageRotationDegrees = image.imageInfo.rotationDegrees
        val detectedObjectList = detect(image.image!!)
        listener(detectedObjectList)
        image.close()
    }

    // YUV -> RGB bitmap -> tensorflow Image -> tensorflow Buffer로 변환하고 추론하여 결과를 list로 출력
    private fun detect(targetImage: Image): List<DetectionObject> {
        val targetBitmap = Bitmap.createBitmap(targetImage.width, targetImage.height, Bitmap.Config.ARGB_8888)
        yuvToRgbConverter.yuvToRgb(targetImage, targetBitmap) // rgbに変換
        tfImageBuffer.load(targetBitmap)
        val tensorImage = tfImageProcessor.process(tfImageBuffer)

        //Run inference on tflite model
        interpreter.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputMap)

        // make a list of inference results
        val detectedObjectList = arrayListOf<DetectionObject>()
        loop@ for (i in 0 until outputDetectionNum[0].toInt()) {
            val score = outputScores[0][i]
            val label = labels[outputLabels[0][i].toInt()]
            val boundingBox = RectF(
                outputBoundingBoxes[0][i][1] * resultViewSize.width,
                outputBoundingBoxes[0][i][0] * resultViewSize.height,
                outputBoundingBoxes[0][i][3] * resultViewSize.width,
                outputBoundingBoxes[0][i][2] * resultViewSize.height
            )

            // Add only those that are greater than the threshold
            if (score >= SCORE_THRESHOLD) {
                detectedObjectList.add(
                    DetectionObject(
                        score = score,
                        label = label,
                        boundingBox = boundingBox
                    )
                )
            } else {
                // detection 결과 점수가 높은 순서대로 정렬. threshold 아래면 loop 종료
                break@loop
            }
        }
        return detectedObjectList.take(4)
    }
}
