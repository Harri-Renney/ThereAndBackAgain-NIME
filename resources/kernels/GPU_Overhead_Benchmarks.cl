__kernel
void cl_000_nullKernel(__global char* empty)
{

}

__kernel
void cl_006_cpymemorykernel(__global char* srcBuffer, __global char* dstBuffer)
{
	uint idx = get_global_id(0);
	
	dstBuffer[idx] = srcBuffer[idx];
}

__kernel
void unidrectional(__global char* srcBuffer, __global char* dstBuffer)
{
    
}

__kernel
void bidirectional(__global char* srcBuffer, __global char* dstBuffer)
{
	
}

__kernel
void singleSample(__global char* srcBuffer, __global char* dstBuffer)
{

}

__kernel
void basicAudioEffect(__global char* srcBuffer, __global char* dstBuffer)
{

}

__kernel
void ftdtCompute(__global float* gridOne, __global float* gridTwo, __global float* gridThree, __global float* boundaryGain, int samplesIndex, __global float* samples, __global float* excitation, int listenerPosition, int excitationPosition, float propagationFactor, float dampingFactor, int rotationIndex)
{
	//Get index for current and neighbouring nodes//
	int ixy = (get_global_id(1)) * get_global_size(0) + get_global_id(0);
	int ixMy = (get_global_id(1)-1) * get_global_size(0) + get_global_id(0);
	int ixPy = (get_global_id(1)+1) * get_global_size(0) + get_global_id(0);
	int ixyM = (get_global_id(1)) * get_global_size(0) + get_global_id(0)-1;
	int ixyP = (get_global_id(1)) * get_global_size(0) + get_global_id(0)+1;
	
	//Determine each buffer in relation to time from a rotation index//
	__global float* nMOne;
	__global float* n;
	__global float* nPOne;
	if(rotationIndex == 0)
	{
		nMOne = gridOne;
		n = gridTwo;
		nPOne = gridThree;
	}
	else if(rotationIndex == 1)
	{
		nMOne = gridTwo;
		n = gridThree;
		nPOne = gridOne;
	}
	else if(rotationIndex == 2)
	{
		nMOne = gridThree;
		n = gridOne;
		nPOne = gridTwo;
	}
	
	//Initalise pressure values//
	float centrePressureNMO = nMOne[ixy];
	float centrePressureN = n[ixy];
	float leftPressure;
	float rightPressure;
	float upPressure;
	float downPressure;

	//Predicate method//
	//leftPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixMy];
    //rightPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixPy];
    //upPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixyM];
    //downPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixyP];

	if(boundaryGain[ixy] > 0.01)
	{
		leftPressure = n[ixy] * boundaryGain[ixy];
		rightPressure = n[ixy] * boundaryGain[ixy];
		upPressure = n[ixy] * boundaryGain[ixy];
		downPressure = n[ixy] * boundaryGain[ixy];
	}
	else
	{
		leftPressure = n[ixMy];
		rightPressure = n[ixPy];
		upPressure = n[ixyM];
		downPressure = n[ixyP];
	}
	
	//Calculate the nex pressure value//
	float newPressure = 2 * centrePressureN;
	newPressure += (dampingFactor - 1.0) * centrePressureNMO;
	newPressure += propagationFactor * (leftPressure + rightPressure + upPressure + downPressure - (4 * centrePressureN));
	newPressure *= 1.0 / (dampingFactor + 1.0);
	
	
	//If the cell is the listener position, sets the next sound sample in buffer to value contained here//
	if(ixy == listenerPosition)
	{
		samples[samplesIndex] = n[ixy];
	}
	
	if(ixy == excitationPosition)	//If the position is an excitation...
	{
		newPressure += excitation[samplesIndex];	//Input excitation value into point. Then increment to next excitation in next iteration.
	}
	
	nPOne[ixy] = newPressure;
}