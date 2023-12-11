using UnityEngine;

public class KeyPointsManager : MonoBehaviour
{
    private KeypointsLoader keyPointsLoader = new KeypointsLoader();

    void Start()
    {
        Debug.Log("IN START");
        keyPointsLoader.LoadKeyPoints();
    }
}
