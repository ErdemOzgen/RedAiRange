package co.uk.deepcyber.imrecs;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.provider.MediaStore;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class TestImages {
    static void copyImagesToGallery(Activity activity) {
        try {
            // Open the zip file from assets
            ZipInputStream zipInputStream = new ZipInputStream(activity.getAssets().open("test_images.zip"));

            ZipEntry zipEntry;
            int successCount = 0;
            int totalCount = 0;
            int imageNo = 1;  // Initialize the image number counter
            while ((zipEntry = zipInputStream.getNextEntry()) != null) {
                // Ensure the entry is not a directory and has a .png or .jpg extension
                if (!zipEntry.isDirectory() &&
                        (zipEntry.getName().endsWith(".png") || zipEntry.getName().endsWith(".jpg"))) {
                    totalCount++;

                    // Read the image data
                    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                    byte[] buffer = new byte[1024];
                    int count;
                    while ((count = zipInputStream.read(buffer)) != -1) {
                        byteArrayOutputStream.write(buffer, 0, count);
                    }
                    byte[] imageData = byteArrayOutputStream.toByteArray();

                    // Convert the image data to a Bitmap
                    Bitmap bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.length);

                    // Get the image title from the file name
                    String imageTitle = zipEntry.getName().substring(0, zipEntry.getName().lastIndexOf('.'));

                    // Create the image description with the image number
                    String imageDescription = "Test Image " + imageNo;

                    // Save the image to the gallery
                    String savedImageURL = MediaStore.Images.Media.insertImage(
                            activity.getContentResolver(),
                            bitmap,
                            imageTitle,
                            imageDescription
                    );

                    if (savedImageURL != null) {
                        successCount++;
                    }

                    // Increment the image number counter for the next image
                    imageNo++;

                    // Close the ByteArrayOutputStream
                    byteArrayOutputStream.close();
                }

                // Close the current ZipEntry
                zipInputStream.closeEntry();
            }

            // Close the ZipInputStream
            zipInputStream.close();

            // Show a toast message after all images have been processed
            Toast.makeText(activity, successCount + " out of " + totalCount + " images saved to gallery", Toast.LENGTH_SHORT).show();

        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(activity, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }


}
