/**
 * desc:
 * author: Benedict R. Gaster
 * copyright: Benedict R. Gaster 2018
 */

#pragma once

class DSP {
private:
protected:
public:
	DSP() {}
	virtual ~DSP() {}

	virtual bool fillBuffer(unsigned long frames, void* inbuf, void* outbuf) = 0;

	virtual float nextSample() = 0;

	virtual int getNumInputs() = 0;
	virtual int getNumOutputs() = 0;
};