__kernel
void cl_000_nullKernel(__global char* empty)
{

}

__kernel
void cl_testkernel(__global float* buffer)
{
	uint idx = get_global_id(0);
	
	buffer[idx] = buffer[idx] * 0.5;
}

__kernel
void cl_006_cpymemorykernel(__global float* srcBuffer, __global float* dstBuffer)
{
	uint idx = get_global_id(0);
	
	dstBuffer[idx] = srcBuffer[idx];
}

__kernel
void cl_007_singlesample(__global float* singleSample)
{
	const float coefficient = 0.5;
	uint idx = get_global_id(0);
	
	singleSample[0] = singleSample[0] * coefficient;
}

__kernel
void cl_008_simplebufferprocessing(__global float* inputBuffer, __global float* outputBuffer)
{
	int idx = get_global_id(0);
	
	float attenuatedSample = inputBuffer[idx] * 0.5;
	//float attenuatedSample = inputBuffer[idx] * pow(M_E, -idx);
	outputBuffer[idx] = attenuatedSample;
}

__kernel
void cl_009_complexbufferprocessing(__global float* inputBuffer, __global float* outputBuffer)
{
	int idx = get_global_id(0);
	int limUpper = get_global_size(0) - 2;
	int limLower = 2;
	
	//float attenuationCoefficient = pow(M_E, -idx);
	
	float smoothedSample = inputBuffer[idx];
	if(idx > limLower & idx < limUpper)
	{
		smoothedSample = ((inputBuffer[idx-2] + 2.0 * inputBuffer[idx-1] + 3.0 * inputBuffer[idx] + 2.0 * inputBuffer[idx+1] + inputBuffer[idx+2]) / 9.0);
	}
	
	//float smoothedSample = idx > limLower & idx < limUpper ? ((inputBuffer[idx-2] + 2.0 * inputBuffer[idx-1] + 3.0 * inputBuffer[idx] + 2.0 * inputBuffer[idx+1] + inputBuffer[idx+2]) / 9.0) : inputBuffer[idx];
	outputBuffer[idx] = smoothedSample;
	//outputBuffer[idx] = smoothedSample * attenuationCoefficient;
}

__kernel 
void cl_010_simplebuffersynthesis(int sampleRate,
								  float frequency,
								  __global float* output)
{
	int global_id = get_global_id(0);
	
	float amplitude = 0.5;
	float relativeFrequency = frequency / (float)sampleRate;
	int time = global_id;
	float currentSample = amplitude * sin(2.0 * M_PI * relativeFrequency * time);
	output[global_id] = currentSample;
}

__kernel
void cl_012_interruptedbufferprocessing(__global float* inputBuffer, __global float* outputBuffer)
{
	int idx = get_global_id(0);
	
	//float attenuatedSample = inputBuffer[idx] * 0.5;
	float attenuatedSample = inputBuffer[idx] * pow(M_E, -idx);
	outputBuffer[idx] = attenuatedSample;
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