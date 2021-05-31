package com.sirslab.nftnn

import android.Manifest
import android.app.Activity
import android.app.PendingIntent.getActivity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.media.ToneGenerator.TONE_CDMA_ABBR_ALERT
import android.os.Bundle
import android.os.Vibrator
import android.support.wearable.activity.WearableActivity
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.wear.widget.BoxInsetLayout
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.*


class MainActivity : WearableActivity(), SensorEventListener, View.OnClickListener {
    private var logTag = "NFTNN"
    private var modelFileName = "tflite_play_092.tflite"
    private var sequenceLength = 79
    private var inferencePeriod = 50
    private var logFileName = "test.csv"

    private var anglesMode = false
    private var pitch_max = -30
    private var pitch_min = -100
    private var roll_min = -90
    private var roll_max = 70
    private var stateDanger = false
    private var lastVibTime:Long = 0
    private lateinit var vibrator: Vibrator
    private lateinit var toneGen: ToneGenerator
    private val tone = ToneGenerator.TONE_PROP_BEEP
    private val vibrationLength = 1000
    private lateinit var tflite: Interpreter
    private lateinit var tfliteModel: MappedByteBuffer
    private var acquisitionMode = false
    private var inferencing = false
    private var acc: Sensor? = null
    private var started = false
    private var file: BufferedWriter? = null
    private var globalBufferLength = 1000
    private var sequenceGlobalBuffer = Array(1) {Array(globalBufferLength) {FloatArray(3) {0.0f} }}
    private var sequenceBuffer = Array(1) {Array(79) {FloatArray(3) {0.0f} }}
    private var sequenceOutput = Array(1) {FloatArray(1)  {0.0f}}
    private var sequenceIdx = 0

    private var thresholdLow = 0.4
    private var thresholdHigh = 0.9

    private lateinit var sensorManager: SensorManager

    private fun updateVibration() {
        val t = System.currentTimeMillis()
        if (stateDanger && (lastVibTime +vibrationLength < t)) {
            vibrator .vibrate(vibrationLength.toLong())
            toneGen.startTone(TONE_CDMA_ABBR_ALERT,vibrationLength)
            lastVibTime = t
        }
    }

    private fun initSensors(){
        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        toneGen = ToneGenerator(AudioManager.STREAM_NOTIFICATION,100)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val permission = ContextCompat.checkSelfPermission(this, Manifest.permission.VIBRATE)
        if (permission != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,arrayOf(Manifest.permission.VIBRATE),1)
        }

        // Get Sensors
        acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        if (acc!= null) {
            Log.d(logTag, "Acc found")
            acc?.also { m ->  sensorManager.registerListener(this, m, 20000)}
        }
        else {
            Log.d(logTag, "Acc NOT found")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        if(!acquisitionMode) {
            tfliteModel = loadModelFile(this)
            tflite = Interpreter(tfliteModel, Interpreter.Options())
        }

        // Enables Always-on
        setAmbientEnabled()

        file = openFile()
        initSensors()

        val context: Context = applicationContext
        var files: Array<String> = context.fileList()

        Log.d(logTag, files.toString())
        Log.d(logTag, context.getExternalFilesDir(null).toString())

    }

    private fun openFile(): BufferedWriter? {
        try {
            val context: Context = applicationContext
            val file = File(context.getExternalFilesDir(null), logFileName.toString())
            if (!file.exists()) {
                file.createNewFile()
            }
            val fw = FileWriter(file.getAbsoluteFile())
            val bw = BufferedWriter(fw)
            return bw
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return null
    }

    private fun writeToCSv(file: BufferedWriter?, s: String){
        Log.d(logTag, "Writing")
        if (file != null) {
            file.write(s)
            file.newLine()
        }
    }

    fun inferenceCallback(x: Float){
        runOnUiThread {
            val textview = findViewById(R.id.textView) as TextView
            textview.setText(String.format("%.2f",x))

            stateDanger = false
            val container = findViewById(R.id.container) as BoxInsetLayout
            var c = Color.BLACK

            if(x > thresholdLow)
                c = Color.YELLOW
            if(x > thresholdHigh) {
                c = Color.RED
                stateDanger = true
            }
            updateVibration()

            container.setBackgroundColor(c)
            container.invalidate()
            Log.d(logTag, "Network output: "+x)
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if(event?.sensor == acc){
            if(!started){
                started = true
                runOnUiThread {
                    val container = findViewById(R.id.container) as BoxInsetLayout
                    container.setBackgroundColor(Color.WHITE)
                    container.invalidate()
                }
            }
            val v = event?.values ?: return

            if(acquisitionMode){
                file?.let { writeToCSv(it, event.timestamp.toString() +","+ v[0].toString() + ","+ v[1].toString() +","+ v[2].toString()) }
            } else {
                updateBuffer(v)
            }
        }
    }

    private fun updateBuffer(v: FloatArray) {
        if(anglesMode){
            val roll = atan2(v[1], v[2]) * 180/kotlin.math.PI;
            val pitch = atan2(-v[0], sqrt(v[1]*v[1] + v[2]*v[2])) * 180/kotlin.math.PI;
            val alert = (pitch > pitch_min && pitch < pitch_max) && (roll > roll_min && roll < roll_max)
            Log.d(logTag, "R: "+roll.toString() + " P: "+pitch.toString()+" A:"+alert.toString())
            if(alert) inferenceCallback(1.0f)
            else inferenceCallback(0.0f)
            return
        }
        if(inferencing) return;
        sequenceGlobalBuffer[0][sequenceIdx][0] = v[0]
        sequenceGlobalBuffer[0][sequenceIdx][1] = v[1]
        sequenceGlobalBuffer[0][sequenceIdx][2] = v[2]

        if((sequenceIdx+1) % inferencePeriod == 0 && (sequenceIdx+1) > sequenceLength){
            var startIdx = sequenceIdx+1-sequenceLength
            for(i in 0..sequenceLength-1){
                sequenceBuffer[0][i][0] = sequenceGlobalBuffer[0][startIdx+i][0]
                sequenceBuffer[0][i][1] = sequenceGlobalBuffer[0][startIdx+i][1]
                sequenceBuffer[0][i][2] = sequenceGlobalBuffer[0][startIdx+i][2]
            }

            inferencing = true
            tflite.run(sequenceBuffer, sequenceOutput)
            inferenceCallback(sequenceOutput[0][0])
            inferencing = false
        }

        sequenceIdx += 1
        if(sequenceIdx == globalBufferLength){
            sequenceIdx = 0
        }
    }

    private fun loadModelFile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        //
    }

    override fun onClick(v: View?) {
        Log.d(logTag, "Close file!")
        file?.close()
        finishAndRemoveTask()
        System.exit(0)
    }
}

