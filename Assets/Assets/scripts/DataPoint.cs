using UnityEngine;

public class DataPoint 
{
    public float[] inputs;
    public float[] expectedOutputs;

   public DataPoint(float[] inputs, float[] expectedOutputs)
    {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }

    public void setInputs(float[] inputs)
    {
        this.inputs = inputs;
    }
}
