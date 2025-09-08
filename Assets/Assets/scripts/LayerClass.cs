using NUnit.Framework;
using System.Collections.Generic;
using UnityEngine;

public class LayerClass 
{
    public int numNodesIn, numNodesOut;
    public float[,] costGradientW;
    public float[] costGradientB;
    public float[,] weights;
    public float[] biases;

    public float[] activations;
    public float[] weightedInputs;

    public LayerClass(int numNodesIn, int numNodesOut)
    {
        costGradientW = new float[numNodesIn, numNodesOut];
        costGradientB = new float[numNodesOut];

        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        weights = new float[numNodesIn, numNodesOut];
        biases = new float[numNodesOut];

        InitializeRandomWeights();

    }

    public float[] CalculateOutputs(float[] inputs)
    {
        activations = new float[numNodesOut];
        weightedInputs = new float[numNodesOut];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            float weightedInput = biases[nodeOut];
            for(int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weightedInput += inputs[nodeIn] * weights[nodeIn, nodeOut];
            }
            activations[nodeOut] = ActivationFunction(weightedInput);
            weightedInputs[nodeOut] = weightedInput;
        }

        
        

        return activations;
    }

    float ActivationFunction(float weightedInput)
    {
        return 1f / (1f + Mathf.Exp(-weightedInput));
    }

    public float ActivationDerivative(float weightedInput)
    {
        float activation = ActivationFunction(weightedInput);
        return activation * (1f - activation);
    }

    public float NodeCost(float outputActivation, float expectedOutput)
    {
        float error = outputActivation - expectedOutput;
        return (error * error);
    }

    float NodeCostDerivative(float outputActivation, float expectedOutput)
    {
        return 2f * (outputActivation - expectedOutput);
    }

    public void ApplyGradients(float learnRate)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            biases[nodeOut] -= costGradientB[nodeOut] * learnRate;
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weights[nodeIn, nodeOut] -= costGradientW[nodeIn, nodeOut] * learnRate;

                Debug.Log("weight " + nodeIn.ToString() + ", " + nodeOut.ToString() + "is " + weights[nodeIn,nodeOut]);
            }
        }
    }

    public void UpdateGradients(float[] inputs, float[] nodeValues)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                float derivativeCostWrtWeight = inputs[nodeIn] * nodeValues[nodeOut];
                //float derivativeCostWrtWeight = nodeValues[nodeOut];

                costGradientW[nodeIn, nodeOut] += derivativeCostWrtWeight;
            }

            float derivativeCostWrtBias = 1f * nodeValues[nodeOut];
            costGradientB[nodeOut] += derivativeCostWrtBias;
        }
    }

    public void ClearGradients()
    {
        costGradientW = new float[numNodesIn, numNodesOut];
        costGradientB = new float[numNodesOut];
    }

    public float[] CalculateOutputLayerNodeValues(float[] expectedOutputs)
    {
        float[] nodeValues = new float[expectedOutputs.Length];

        for (int i = 0; i < nodeValues.Length; i++)
        {
            float costDerivateive = NodeCostDerivative(activations[i], expectedOutputs[i]);
            float activationDerivative = ActivationDerivative(weightedInputs[i]);
            nodeValues[i] = activationDerivative * costDerivateive;
        }

        return nodeValues;
    }

    public void InitializeRandomWeights()
    {
        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
        {
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                float randomValue = Random.Range(-1f, 1f);
                weights[nodeIn, nodeOut] = randomValue / Mathf.Sqrt(numNodesIn);
            }
        }
    }

    public float[] CalculateHiddenLayerNodeValues(LayerClass oldLayer, float[] oldNodeValues)
    {
        float[] newNodeValues = new float[numNodesOut];

        for (int newNodeIndex = 0; newNodeIndex < newNodeValues.Length; newNodeIndex++)
        {
            float newNodeValue = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.Length; oldNodeIndex++)
            {
                float weightedInputDerivative = oldLayer.weights[newNodeIndex, oldNodeIndex];
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
            }
            newNodeValue *= ActivationDerivative(weightedInputs[newNodeIndex]);
            newNodeValues[newNodeIndex] = newNodeValue;
        }

        return newNodeValues;

    }
}
