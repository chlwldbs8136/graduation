package com.example.objectdetectionapp

import android.graphics.RectF

/**
 * detection 결과 class
 */
data class DetectionObject(
    val score: Float,
    val label: String,
    val boundingBox: RectF
)