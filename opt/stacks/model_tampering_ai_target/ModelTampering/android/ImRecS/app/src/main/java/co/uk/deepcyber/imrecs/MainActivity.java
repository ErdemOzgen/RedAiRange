package co.uk.deepcyber.imrecs;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import org.tensorflow.lite.Interpreter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.FileInputStream;
import java.util.Arrays;

import android.content.res.AssetFileDescriptor;
import android.graphics.drawable.BitmapDrawable;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import android.util.Log;
import android.Manifest;


public class MainActivity extends AppCompatActivity {
    private static final String[] CIFAR10_CLASS_NAMES = {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
    };
    private static final int CAMERA_REQUEST = 1888;
    private static final int GALLERY_REQUEST = 1000;
    ImageView imageView;
    TextView classificationResultTextView;

    Interpreter interpreter;
    int NUM_CLASSES = 10;
    private static final int REQUEST_STORAGE_PERMISSION = 1;  // Define the constant here

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        this.imageView = (ImageView) this.findViewById(R.id.imageView);
        this.classificationResultTextView = findViewById(R.id.classificationResultTextView);

        Button testImgButton = (Button) this.findViewById(R.id.button0);
        Button photoButton = (Button) this.findViewById(R.id.button1);
        Button galleryButton = (Button) this.findViewById(R.id.button2);
        TestImages.copyImagesToGallery(this);

        TestImages.copyImagesToGallery(this);
        // Load TFLite model
        try {
            interpreter = new Interpreter(loadModelFile());
        } catch (Exception e) {
            e.printStackTrace();
        }

        testImgButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imageView.setImageResource(R.drawable.test_image);
                classify(imageView);
            }
        });



        photoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            }
        });

        galleryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, GALLERY_REQUEST);
            }
        });
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_REQUEST && resultCode == RESULT_OK) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(photo);
            classify(imageView);
        } else if (requestCode == GALLERY_REQUEST && resultCode == RESULT_OK) {
            Uri selectedImage = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                imageView.setImageBitmap(bitmap);
                classify(imageView);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_STORAGE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                TestImages.copyImagesToGallery(this);
            } else {
                Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("simple-cifar10.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    public String classify(ImageView imageView) {
        // Get the image from the ImageView
        Bitmap bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();

        // Resize the image to 32x32
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 32, 32, true);

        // Convert the resized image to a ByteBuffer
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 32 * 32 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int pixel = resizedBitmap.getPixel(x, y);
                byteBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((pixel & 0xFF) / 255.0f);
            }
        }

        try {
            TensorBuffer inputBuffer = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            inputBuffer.loadBuffer(byteBuffer);
            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.FLOAT32);
            interpreter.run(inputBuffer.getBuffer(), outputBuffer.getBuffer());
            int labelIndex = getMaxIndex(outputBuffer.getFloatArray());
            String className = CIFAR10_CLASS_NAMES[labelIndex];
            classificationResultTextView.setText("Image was classified as "+className);
            // Log the classification result
            Log.d("MainActivity", "Classification result: " + className);
            return className;
        } catch (Exception e) {
            e.printStackTrace();
            return "Error";
        }
    }

    private int getMaxIndex(float[] probabilities) {
        int maxIndex = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    private String classify2(ImageView imageView) {
        // Get the image from the ImageView
            Bitmap bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();

            // Resize the image to 32x32
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 32, 32, true);

        // Convert the resized bitmap to a float array
        int width = resizedBitmap.getWidth();
        int height = resizedBitmap.getHeight();
        int[] pixels = new int[width * height];
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        float[] floatArray = new float[pixels.length * 3];
        for (int i = 0; i < pixels.length; ++i) {
            int pixel = pixels[i];
            floatArray[i * 3] = (pixel >> 16) & 0xff;
            floatArray[i * 3 + 1] = (pixel >> 8) & 0xff;
            floatArray[i * 3 + 2] = pixel & 0xff;
        }

        // Prepare the model input
        TensorBuffer inputBuffer = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
        inputBuffer.loadArray(floatArray);

        // Prepare the model output
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, NUM_CLASSES}, DataType.FLOAT32);

        // Run inference
        interpreter.run(inputBuffer.getBuffer(), outputBuffer.getBuffer());

        // Process the model output
        TensorProcessor tensorProcessor =
                new TensorProcessor.Builder()
                        .add(getPostprocessNormalizeOp())
                        .build();

        TensorBuffer processedOutputBuffer = tensorProcessor.process(outputBuffer);

        // Get the classification result (assuming the result is a string label)
        String classificationResult = getClassificationResult(processedOutputBuffer);
        classificationResultTextView.setText(classificationResult);
        // Log the classification result
        Log.d("MainActivity", "Classification result: " + classificationResult);

        return classificationResult;
    }

    private TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(0, 1);
    }

    private String getClassificationResult(TensorBuffer outputBuffer) {
        // Assume the model's output is a single integer representing the class index.
        // This code gets the maximum value's index from the output buffer, which corresponds to the most likely class index.
        int classIndex = outputBuffer.getIntArray()[0];
        return CIFAR10_CLASS_NAMES[classIndex];
    }

}
