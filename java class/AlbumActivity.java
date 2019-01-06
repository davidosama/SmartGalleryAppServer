package com.example.android.camera2basic;

import android.app.Activity;
import android.content.Intent;
import android.content.res.Resources;
import android.database.Cursor;
import android.database.MergeCursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.SparseIntArray;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.bumptech.glide.Glide;

import org.json.JSONObject;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class AlbumActivity extends AppCompatActivity {

    GridView galleryGridView;
    ArrayList<HashMap<String, String>> imageList = new ArrayList<HashMap<String, String>>();
    String album_name = "";
    LoadAlbumImages loadAlbumTask;
    static String CaptionFromServer="";
    String URI_To_Server="http://192.168.1.6:5000/api";


    //caption variables
    private static final String MODEL_FILE = "file:///android_asset/merged_frozen_graph.pb";
    private static final String INPUT1 = "encoder/import/InputImage:0";
    private static final String OUTPUT_NODES = "DecoderOutputs.txt";
    private static final int NUM_TIMESTEPS = 22;
    private static final int IMAGE_SIZE = 299;
    private static final int IMAGE_CHANNELS = 3;
    private static final int[] DIM_IMAGE = new int[]{1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS};
    private TensorFlowInferenceInterface inferenceInterface;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private String[] OutputNodes = null;
    private String[] WORD_MAP = null;
    Uri uri;
    /**
     * Conversion from screen rotation to JPEG orientation.
     */
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    private static final int REQUEST_CAMERA_PERMISSION = 1;
    private static final String FRAGMENT_DIALOG = "dialog";

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_album);
        inferenceInterface = InitSession();
        Intent intent = getIntent();
        album_name = intent.getStringExtra("name");
        setTitle(album_name);


        galleryGridView = (GridView) findViewById(R.id.galleryGridView);
        int iDisplayWidth = getResources().getDisplayMetrics().widthPixels;
        Resources resources = getApplicationContext().getResources();
        DisplayMetrics metrics = resources.getDisplayMetrics();
        float dp = iDisplayWidth / (metrics.densityDpi / 160f);

        if (dp < 360) {
            dp = (dp - 17) / 2;
            float px = Function.convertDpToPixel(dp, getApplicationContext());
            galleryGridView.setColumnWidth(Math.round(px));
        }


        loadAlbumTask = new LoadAlbumImages();
        loadAlbumTask.execute();


    }

    public String runModel(Bitmap imBitmap) {
        return GenerateCaptions(Preprocess(imBitmap));
    }


    String[] LoadFile(String fileName) {
        InputStream is = null;
        try {
            is = this.getAssets().open(fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
        BufferedReader r = new BufferedReader(new InputStreamReader(is));
        StringBuilder total = new StringBuilder();
        String line;
        try {
            while ((line = r.readLine()) != null) {
                total.append(line).append('\n');
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return total.toString().split("\n");
    }

    TensorFlowInferenceInterface InitSession() {
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(this.getAssets(), MODEL_FILE);
        OutputNodes = LoadFile(OUTPUT_NODES);
        WORD_MAP = LoadFile("idmap");
        return inferenceInterface;
    }

    String GenerateCaptions(float[] imRGBMatrix) {
        inferenceInterface.fillNodeFloat(INPUT1, DIM_IMAGE, imRGBMatrix);
        inferenceInterface.runInference(OutputNodes);

        String result = "";
        int temp[][] = new int[NUM_TIMESTEPS][1];
        for (int i = 0; i < NUM_TIMESTEPS; ++i) {
            inferenceInterface.readNodeInt(OutputNodes[i], temp[i]);
            if (temp[i][0] == 2/*</S>*/) {
                return result;
            }
            result += WORD_MAP[temp[i][0]] + " ";
        }
        return null;
    }

    float[] Preprocess(Bitmap imBitmap) {
        imBitmap = Bitmap.createScaledBitmap(imBitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        float[] floatValues = new float[IMAGE_SIZE * IMAGE_SIZE * 3];

        imBitmap.getPixels(intValues, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((float) ((val >> 16) & 0xFF)) / 255;//R
            floatValues[i * 3 + 1] = ((float) ((val >> 8) & 0xFF)) / 255;//G
            floatValues[i * 3 + 2] = ((float) ((val & 0xFF))) / 255;//B
        }
        return floatValues;
    }


    class LoadAlbumImages extends AsyncTask<String, Void, String> {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            imageList.clear();
        }

        protected String doInBackground(String... args) {
            String xml = "";

            String path = null;
            String album = null;
            String timestamp = null;
            Uri uriExternal = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
            Uri uriInternal = MediaStore.Images.Media.INTERNAL_CONTENT_URI;

            String[] projection = {MediaStore.MediaColumns.DATA,
                    MediaStore.Images.Media.BUCKET_DISPLAY_NAME, MediaStore.MediaColumns.DATE_MODIFIED};

            Cursor cursorExternal = getContentResolver().query(uriExternal, projection, "bucket_display_name = \"" + album_name + "\"", null, null);
            Cursor cursorInternal = getContentResolver().query(uriInternal, projection, "bucket_display_name = \"" + album_name + "\"", null, null);
            Cursor cursor = new MergeCursor(new Cursor[]{cursorExternal, cursorInternal});
            while (cursor.moveToNext()) {

                path = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DATA));
                album = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.BUCKET_DISPLAY_NAME));
                timestamp = cursor.getString(cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DATE_MODIFIED));

                imageList.add(Function.mappingInbox(album, path, timestamp, Function.converToTime(timestamp), null));
            }
            cursor.close();
            Collections.sort(imageList, new MapComparator(Function.KEY_TIMESTAMP, "dsc")); // Arranging photo album by timestamp decending
            return xml;
        }

        @Override
        protected void onPostExecute(String xml) {

            SingleAlbumAdapter adapter = new SingleAlbumAdapter(AlbumActivity.this, imageList);
            galleryGridView.setAdapter(adapter);
            galleryGridView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                public void onItemClick(AdapterView<?> parent, View view,
                                        final int position, long id) {
                    Intent intent = new Intent(AlbumActivity.this, GalleryPreview.class);
                    intent.putExtra("path", imageList.get(+position).get(Function.KEY_PATH));
                    startActivity(intent);
                }
            });
        }
    }


    class SingleAlbumAdapter extends BaseAdapter {
        private Activity activity;
        private ArrayList<HashMap<String, String>> data;

        public SingleAlbumAdapter(Activity a, ArrayList<HashMap<String, String>> d) {
            activity = a;
            data = d;
        }

        public int getCount() {
            return data.size();
        }

        public Object getItem(int position) {
            return position;
        }

        public long getItemId(int position) {
            return position;
        }

        public View getView(int position, View convertView, ViewGroup parent) {
            SingleAlbumViewHolder holder = null;
            if (convertView == null) {
                holder = new SingleAlbumViewHolder();
                convertView = LayoutInflater.from(activity).inflate(
                        R.layout.single_album_row, parent, false);

                holder.galleryImage = (ImageView) convertView.findViewById(R.id.galleryImage);
                holder.caption = (TextView) convertView.findViewById(R.id.caption);

                convertView.setTag(holder);
            } else {
                holder = (SingleAlbumViewHolder) convertView.getTag();
            }
            holder.galleryImage.setId(position);
            holder.caption.setId(position);

            final HashMap<String, String> song = data.get(position);
            try {
                holder.galleryImage.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        JSONObject postData = new JSONObject();
                        try{
                            postData.put("image", getStringFromBitmap(getBitmap(song.get(Function.KEY_PATH))));
                            Log.i("MINA","MINA "+postData.toString());
                            new SendDeviceDetails().execute(URI_To_Server, postData.toString());
                            Toast.makeText(activity, AlbumActivity.CaptionFromServer, Toast.LENGTH_SHORT).show();
                        }
                        catch (Exception e){
                            e.printStackTrace();
                        }

//                        Toast.makeText(activity, runModel(getBitmap(song.get(Function.KEY_PATH))), Toast.LENGTH_SHORT).show();

                    }
                });
                Glide.with(activity)
                        .load(new File(song.get(Function.KEY_PATH))) // Uri of the picture
                        .into(holder.galleryImage);



            } catch (Exception e) {
            }
            return convertView;
        }
        private String getStringFromBitmap(Bitmap bitmapPicture) {
            /*
             * This functions converts Bitmap picture to a string which can be
             * JSONified.
             * */
            final int COMPRESSION_QUALITY = 100;
            String encodedImage;
            ByteArrayOutputStream byteArrayBitmapStream = new ByteArrayOutputStream();
            bitmapPicture.compress(Bitmap.CompressFormat.PNG, COMPRESSION_QUALITY,
                    byteArrayBitmapStream);
            byte[] b = byteArrayBitmapStream.toByteArray();
            encodedImage = Base64.encodeToString(b, Base64.DEFAULT);
            return encodedImage;
        }

        public Bitmap getBitmap(String path) {
            try {
                Bitmap bitmap = null;
                File f = new File(path);
                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inPreferredConfig = Bitmap.Config.ARGB_8888;

                bitmap = BitmapFactory.decodeStream(new FileInputStream(f), null, options);
                return bitmap;
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }
    }


    class SingleAlbumViewHolder {
        ImageView galleryImage;
        TextView caption;
    }

//    private static final int RequestPermissionCode = 1;
//

//    TextView CaptionTxt ;
//    Button btn;
//    ImageView imageView;
//
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_main);
//
//        CaptionTxt = findViewById(R.id.Caption_text);
//        btn = findViewById(R.id.CaptureTxt);
//        imageView = findViewById(R.id.imageview);
//        btn.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                openCamAndCrop();
//            }
//        });
//
//
//
//    }
//    private void openCamAndCrop()
//    {
//        int permissionCheck = ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA);
//        if(permissionCheck == PackageManager.PERMISSION_GRANTED)
//            CameraOpen();
//        else
//            RequestRuntimePermission();
//
//    }
//
//    private void RequestRuntimePermission() {
//        if(ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this,Manifest.permission.CAMERA))
//            Toast.makeText(this,"CAMERA permission allows us to access CAMERA app",Toast.LENGTH_SHORT).show();
//        else
//        {
//            ActivityCompat.requestPermissions(MainActivity.this,new String[]{Manifest.permission.CAMERA},RequestPermissionCode);
//        }
//    }
//    private void CameraOpen() {
//
//
//        CropImage.activity()
//                .setGuidelines(CropImageView.Guidelines.ON)
//                .start(this);
//
//    }
//    private void getTextFromImage(Bitmap bitmap)
//    {
//        CaptionTxt.setText(getString(R.string.caption_will_be_shown_here));
//        Toast.makeText(MainActivity.this,"Please wait while generating Caption",Toast.LENGTH_LONG).show();
//        imageView.setImageBitmap(bitmap);
//        new GenerateCaptionTask().execute(bitmap);
//
//
//    }
//
//    @Override
//    protected void onStart() {
//        super.onStart();
//
//    }
//


//

//

//
//    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
//        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
//            CropImage.ActivityResult result = CropImage.getActivityResult(data);
//            if (resultCode == RESULT_OK) {
//                uri = result.getUri();
////                imageView.setImageURI(null);
////                imageView.setImageURI(uri);
//                try {
//                    getTextFromImage(MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri));
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//            }
//        }
//    }
//
//    private class GenerateCaptionTask extends AsyncTask<Bitmap,Void,String> {
//        @Override
//        protected String doInBackground(Bitmap... bitmaps) {
//            final String text = runModel(bitmaps[0]);
//            return text;
//        }
//
//        // Before the tasks execution
//        protected void onPreExecute(){
//            CaptionTxt.setText("Please wait while generating caption");
//        }
//        protected void onPostExecute(String result){
//            CaptionTxt.setText(result);
//
//        }
//    }
//
//
//}

    private class SendDeviceDetails extends AsyncTask<String, Void, String> {

        @Override
        protected String doInBackground(String... params) {
            AlbumActivity.CaptionFromServer="";
            String data = "";

            HttpURLConnection httpURLConnection = null;
            try {
                httpURLConnection = (HttpURLConnection) new URL(params[0]).openConnection();
                httpURLConnection.setRequestProperty("Content-Type","application/json; charset=UTF-8");
                httpURLConnection.setRequestMethod("POST");

                httpURLConnection.setDoOutput(true);

                DataOutputStream wr = new DataOutputStream(httpURLConnection.getOutputStream());
                wr.writeBytes(params[1]);
                wr.flush();
                wr.close();

                InputStream in = httpURLConnection.getInputStream();
                InputStreamReader inputStreamReader = new InputStreamReader(in);

                int inputStreamData = inputStreamReader.read();
                while (inputStreamData != -1) {
                    char current = (char) inputStreamData;
                    inputStreamData = inputStreamReader.read();
                    data += current;
                }
                AlbumActivity.CaptionFromServer=data;
            } catch (Exception e) {
                Log.i("MINA","EXCEPTION IN THE ASYNC TASK "+e.getMessage()+"\nTO STRING "+e.toString());
                e.printStackTrace();
            } finally {
                if (httpURLConnection != null) {
                    httpURLConnection.disconnect();
                }
            }

            return data;
        }

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            Log.e("TAG", result); // this is expecting a response code to be sent from your server upon receiving the POST data
        }
    }
}