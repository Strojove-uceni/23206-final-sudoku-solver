package com.example.sudokusolverapp;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.content.Intent;
import android.provider.MediaStore;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;

public class MainActivity extends AppCompatActivity {

    private static final int pic_id = 123;
    private static final int CAMERA_PERMISSION_CODE = 1;
    Button photo_btn;
    ImageView photo_image_view;
    String imageString = "";

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        photo_btn = findViewById(R.id.photo_btn);
        photo_image_view = findViewById(R.id.photo_image_view);

        photo_btn.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v)
            {
                if(ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
                {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
                }
                else
                {
                    Intent camera_intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(camera_intent, pic_id);
                }
            }
        });
    }
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == pic_id && data != null)
        {
            Bitmap photo = (Bitmap) data.getExtras().get("data");

            if (!Python.isStarted())
                Python.start(new AndroidPlatform(getApplicationContext()));
            final Python py = Python.getInstance();

            imageString = getImageString(photo);

            PyObject pyObj = py.getModule("imageProcessing");
            PyObject obj = pyObj.callAttr("main", imageString);

            String str = obj.toString();

            byte[] image_data = android.util.Base64.decode(str, Base64.DEFAULT);

            Bitmap bmp = BitmapFactory.decodeByteArray(image_data, 0, image_data.length);

            photo_image_view.setImageBitmap(bmp);
        }
    }

    private String getImageString(Bitmap bitmap)
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG,100,baos);
        byte[] imageBytes = baos.toByteArray();

        String encodedImage = android.util.Base64.encodeToString(imageBytes, Base64.DEFAULT);
        return encodedImage;
    }
}