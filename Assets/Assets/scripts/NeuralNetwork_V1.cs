using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.U2D;

public class NeuralNetwork_V1 : MonoBehaviour
{
    [SerializeField]
    private Camera mainCamera;

    [SerializeField]
    private GameObject square,background;

    [SerializeField]
    private float resolution = 1;

    [SerializeField]
    private Color safeColor, dangerColor, secoundSafeColor;

    [SerializeField]
    float learnRate;

    [SerializeField]
    private DataPoint[] data;

    private bool visualizerIsOn = false;

    [SerializeField]
    GameObject[] dataPoints;

    private NeuralNetworkClass network;
    

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        network = new NeuralNetworkClass(2,5,2);
         
        data = new DataPoint[dataPoints.Length];
        

        for (int i = 0; i < dataPoints.Length; i++)
        {
            
            float[] inputs = new float[2];
            inputs[0] = dataPoints[i].transform.position.x;
            inputs[1] = dataPoints[i].transform.position.y;
           

           

            float[] expectedOutputs = new float[2]; 
            if (dataPoints[i].gameObject.GetComponent<SpriteRenderer>().color == safeColor || dataPoints[i].gameObject.GetComponent<SpriteRenderer>().color == secoundSafeColor)
            {
                expectedOutputs[0] = 1;
                expectedOutputs[1] = 0;
                //Debug.Log("Blue");
            }
            else
            {
                expectedOutputs[0] = 0;
                expectedOutputs[1] = 1;
                
            }

            data[i] = new DataPoint(inputs, expectedOutputs);
            
        }
    }

    // Update is called once per frame
    void Update()
    {
        if ((visualizerIsOn))
        {
            Visualize();
            network.Learn(data, learnRate);
        }
       
        
    }


    public void Visualize()
    {
        float cameraHeight, cameraWidth, stepSize;
        cameraHeight = mainCamera.orthographicSize * 2;
        cameraWidth = mainCamera.orthographicSize * 2 * mainCamera.aspect;
        stepSize = (1 / resolution);

        foreach(Transform child in background.transform)
        {
            Destroy(child.gameObject);
        }

        for (float i = 0; i < cameraHeight; i = i + stepSize) // i is the y cord
        {
            for (float j = 0; j < cameraWidth; j = j + stepSize) // j is the x cord
            {

                GameObject sq = Instantiate(square, new Vector3(j - (cameraWidth / 2f) + mainCamera.transform.position.x + (((square.transform.localScale.x) / resolution) / 2f), i - (cameraHeight / 2f) + mainCamera.transform.position.y + ((square.transform.localScale.y / resolution) / 2f), 0), Quaternion.identity, background.transform);
                sq.transform.localScale = new Vector3(square.transform.localScale.x / resolution, square.transform.localScale.y / resolution, 1);
                float[] inputs = new float[2];
                inputs[0] = sq.gameObject.transform.position.x;
                inputs[1] = sq.gameObject.transform.position.y;
                if (network.Classify(inputs) == 0)
                {
                    sq.GetComponent<SpriteRenderer>().color = safeColor;
                }
                else
                {
                    sq.GetComponent<SpriteRenderer>().color = dangerColor;
                }

                sq.gameObject.name = "Squrae " + j + " " + i;
            }
        }

    }


    public void StartVisualization()
    {
        visualizerIsOn = true;
    }

}
