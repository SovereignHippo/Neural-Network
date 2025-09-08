using NUnit.Framework;
using System.Runtime.InteropServices.WindowsRuntime;
using UnityEngine;
using UnityEngine.Windows;
using UnityEngine.WSA;

public class NeuralNetworkClass
{
    LayerClass[] layers;

    public NeuralNetworkClass(params int[] layerSizes)
    {
        layers = new LayerClass[layerSizes.Length - 1];
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i] = new LayerClass(layerSizes[i], layerSizes[i + 1]);
        }
    }

    float[] CalculateOutputs(float[] inputs)
    {
        foreach (LayerClass layer in layers)
        {
            inputs = layer.CalculateOutputs(inputs);
        }
        return inputs;
    }

    public int Classify(float[] inputs)
    {
        float[] outputs = CalculateOutputs(inputs);
        return IndexOfMaxValue(outputs);
    }

    public int IndexOfMaxValue(float[] outputs)
    {
        float highestValue = 0;
        int indexofHighestValue = 0;
        for (int i = 0; i < outputs.Length; i++)
        {
            if (outputs[i] > highestValue)
            {
                highestValue = outputs[i];
                indexofHighestValue = i;
            }
        }
        return indexofHighestValue;
    }

    float Cost(DataPoint dataPoint)
    {
        float[] outputs = CalculateOutputs(dataPoint.inputs);
        LayerClass outputLayer = layers[layers.Length - 1];
        float cost = 0;

        for (int nodeOut = 0; nodeOut < outputs.Length; nodeOut++)
        {
            cost += outputLayer.NodeCost(outputs[nodeOut], dataPoint.expectedOutputs[nodeOut]);
        }
        Debug.Log("Cost: " + cost);
        return cost;
    }

    float Cost(DataPoint[] data)
    {
        float totalCost = 0;

        foreach (DataPoint dataPoint in data)
        {
            totalCost += Cost(dataPoint);
        }

        return totalCost / data.Length;

    }
    //old learn
    //public void Learn(DataPoint[] trainingData, float learnRate)
    //{
    //    const float h = .0001f;
    //    float originalCost = Cost(trainingData);

    //    foreach (LayerClass layer in layers)
    //    {
    //        for (int nodeIn = 0; nodeIn < layer.numNodesIn; nodeIn++)
    //        {
    //            for (int nodeOut = 0; nodeOut < layer.numNodesOut; nodeOut++)
    //            {
    //                layer.weights[nodeIn, nodeOut] += h;
    //                float deltaCost = Cost(trainingData) - originalCost;
    //                //Debug.Log("Delta = " + deltaCost.ToString());
    //                layer.weights[nodeIn, nodeOut] -= h;
    //                layer.costGradientW[nodeIn, nodeOut] = deltaCost / h;
    //            }
    //        }

    //        for (int biasIndex = 0; biasIndex < layer.biases.Length; biasIndex++)
    //        {
    //            layer.biases[biasIndex] += h;
    //            float deltaCost = Cost(trainingData) - originalCost;
    //            layer.biases[biasIndex] -= h;
    //            layer.costGradientB[biasIndex] = deltaCost / h;
    //        }
    //    }

    //    ApplyAllGradients(learnRate);

    //}

    public void Learn(DataPoint[] trainingBatch, float learnRate)
    {
        foreach (DataPoint datapoint in trainingBatch)
        {
            UpdateAllGradients(datapoint);
        }

        ApplyAllGradients(learnRate / trainingBatch.Length);

        ClearAllGradients();
    }

    void ApplyAllGradients(float learnRate)
    {
        foreach (LayerClass layer in layers)
        {
            layer.ApplyGradients(learnRate);
        }
    }

    void UpdateAllGradients(DataPoint dataPoint)
    {
        CalculateOutputs(dataPoint.inputs);

        LayerClass outputLayer = layers[layers.Length - 1];
        float[] nodeValues = outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs);
        outputLayer.UpdateGradients(layers[layers.Length - 2].activations, nodeValues);

        for (int hiddenLayerIndex = layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            LayerClass hiddenLayer = layers[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
            hiddenLayer.UpdateGradients(layers[hiddenLayerIndex].activations, nodeValues);
        }

    }

    void ClearAllGradients()
    {
        foreach (LayerClass layer in layers)
        {
            layer.ClearGradients();
        }
    }

    

}
