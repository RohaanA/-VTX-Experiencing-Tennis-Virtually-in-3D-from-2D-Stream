using UnityEngine;
using System.Collections.Generic;

[System.Serializable]
public class BoxData
{
    public float x1, y1, x2, y2;
}

[System.Serializable]
public class Keypoints
{
    public List<float> x;
    public List<float> y;
    public List<float> visible;
}

[System.Serializable]
public class KeypointData
{
    public string name;
    public int classId;
    public float confidence;
    public BoxData box;
    public Keypoints keypoints;
}

[System.Serializable]
public class KeypointDataArray
{
    public List<KeypointData> keypoints; // Adjusted to handle the outer array
}

public class KeypointsLoader 
{
    public string dataFolderPath = "D:\\keypoint_data";

    void Start()
    {
        LoadKeyPoints();
    }

    public void LoadKeyPoints()
    {
        string[] files = System.IO.Directory.GetFiles(dataFolderPath, "*.*");

        foreach (var filePath in files)
        {
            string jsonData = System.IO.File.ReadAllText(filePath);

            // Adjusted to handle the outer array
            KeypointDataArray keypointDataArray = JsonUtility.FromJson<KeypointDataArray>("{\"keypoints\":" + jsonData + "}");

            foreach (var frameData in keypointDataArray.keypoints)
            {
                int classId = frameData.classId;
                BoxData playerBoundingBox = frameData.box;
                Keypoints playerKeypoints = frameData.keypoints;

                Debug.Log($"Player classId: {classId} loaded. Keypoints: {playerKeypoints.x.Count}");
            }
            Debug.Log($"Frame: {filePath}");
        }
    }
}
