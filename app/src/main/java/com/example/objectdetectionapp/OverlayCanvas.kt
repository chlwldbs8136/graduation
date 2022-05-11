package com.example.objectdetectionapp

import android.graphics.*
import android.view.SurfaceHolder
import android.view.SurfaceView

/**
 * detection 결과 표시하는 transparent surfaceView
 */
class OverlaySurfaceView(surfaceView: SurfaceView) :
    SurfaceView(surfaceView.context), SurfaceHolder.Callback {

    init {
        surfaceView.holder.addCallback(this)
        surfaceView.setZOrderOnTop(true)
    }

    private var surfaceHolder = surfaceView.holder
    private val paint = Paint()
    private val pathColorList = listOf(Color.RED, Color.GREEN, Color.CYAN, Color.BLUE)

    override fun surfaceCreated(holder: SurfaceHolder) {
        // make surfaceView transparent
        surfaceHolder.setFormat(PixelFormat.TRANSPARENT)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
    }

    /**
     * Display object detection results in surfaceView
     */
    fun draw(detectedObjectList: List<DetectionObject>) {
        // surface holder로 canvas 획득(Stop시에도 draw되어 exception 발생 가능성이 있으므로 nullable 하고 이하 같은 취급)
        val canvas: Canvas? = surfaceHolder.lockCanvas()
        // clar previous drawing
        canvas?.drawColor(0, PorterDuff.Mode.CLEAR)

        detectedObjectList.mapIndexed { i, detectionObject ->
            // viewing the bounding box
            paint.apply {
                color = pathColorList[i]
                style = Paint.Style.STROKE
                strokeWidth = 7f
                isAntiAlias = false
            }
            canvas?.drawRect(detectionObject.boundingBox, paint)

            // view labels and scores
            paint.apply {
                style = Paint.Style.FILL
                isAntiAlias = true
                textSize = 77f
            }
            canvas?.drawText(
                detectionObject.label + " " + "%,.2f".format(detectionObject.score * 100) + "%",
                detectionObject.boundingBox.left,
                detectionObject.boundingBox.top - 5f,
                paint
            )
        }

        surfaceHolder.unlockCanvasAndPost(canvas ?: return)
    }
}
