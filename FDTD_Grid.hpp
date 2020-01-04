#ifndef FDTD_GRID_HPP
#define FDTD_GRID_HPP

#include <iostream>
#include <tuple>

#include "Cartisian_Grid.hpp"

class Model {
private:
	typedef float base_type_;
	//typedef double base_type_;
	typedef Cartisian_Grid<base_type_> GridType_;
	const unsigned int width_;
	const unsigned int height_;
	const unsigned int size_;
	float boundaryGain_;

	GridType_ pressureGrid0_;
	GridType_ pressureGrid1_;
	GridType_ pressureGrid2_;
	GridType_ boundaryGrid_;
	std::tuple<GridType_ *, GridType_ *, GridType_ *> grids_;

	int excitationPosition[2];
	int listenerPosition[2];
public:
	Model()
		: width_(16),
		height_(16),
		size_(width_*height_),
		pressureGrid0_(width_, height_),
		pressureGrid1_(width_, height_),
		pressureGrid2_(width_, height_),
		boundaryGrid_(width_, height_)
	{}
	Model(unsigned int aWidth, unsigned int aHeight, float aBoundaryGain)
		: width_(aWidth),
		height_(aHeight),
		size_(width_*height_),
		boundaryGain_(aBoundaryGain),
		pressureGrid0_(width_, height_),
		pressureGrid1_(width_, height_),
		pressureGrid2_(width_, height_),
		boundaryGrid_(width_, height_)
	{
		grids_ = std::make_tuple((&pressureGrid0_), (&pressureGrid1_), (&pressureGrid2_));

		//Initalise default pressure values//
		//Later may want to add padding to create grid divisible by x16 on each dimension for memory coalescing//
		for (int x = 0; x != width_; ++x)
		{
			for (int y = 0; y != height_; ++y)
			{
				pressureGrid0_.valueAt(x, y) = 0.0;
				pressureGrid1_.valueAt(x, y) = 0.0;
				pressureGrid2_.valueAt(x, y) = 0.0;
				if (isEdge(x, y))
				{
					boundaryGrid_.valueAt(x, y) = boundaryGain_;
				}
				else
				{
					boundaryGrid_.valueAt(x, y) = 0.0;
				}
			}
		}
	}

	bool isEdge(int x, int y) const {
		return (
			x == 0 || y == 0 || x == width_ - 1 || y == height_ - 1);
	}
	GridType_* nMinusOneGrid() const {
		return std::get<0>(grids_);
	}

	GridType_* nGrid() const {
		return std::get<1>(grids_);
	}

	GridType_* nPlusOneGrid() const {
		return std::get<2>(grids_);
	}

	GridType_ boundaryGrid() const {
		return boundaryGrid_;
	}

	void rotateGrids() {
		GridType_* nMinusOne = std::get<0>(grids_);
		GridType_* n = std::get<1>(grids_);
		GridType_* nPlusOne = std::get<2>(grids_);

		std::get<0>(grids_) = n;
		std::get<1>(grids_) = nPlusOne;
		std::get<2>(grids_) = nMinusOne;
	}

	//Get grids as 1 Dimensional buffer - For passing into OpenCL functions//
	base_type_* getNMinusOneGridBuffer()
	{
		return nMinusOneGrid()->getGrid();
	}
	base_type_* getNGridBuffer()
	{
		return nGrid()->getGrid();
	}
	base_type_* getNPlusOneGridBuffer()
	{
		return nPlusOneGrid()->getGrid();
	}
	base_type_* getBoundaryGridBuffer()
	{
		return boundaryGrid_.getGrid();
	}

	//Set positions within grid - Should add boundary checks for legal positions//
	void setExcitationPosition(int x, int y)
	{
		excitationPosition[0] = x;
		excitationPosition[1] = y;
	}
	void setListenerPosition(int x, int y)
	{
		listenerPosition[0] = x;
		listenerPosition[1] = y;
	}
	int getExcitationPosition()
	{
		return (excitationPosition[1] * width_ + excitationPosition[0]);
	}
	int getListenerPosition()
	{
		return (listenerPosition[1] * width_ + listenerPosition[0]);
	}

	//Get the pressure value at the defined listener position//
	base_type_ getSample()
	{
		return nGrid()->valueAt(listenerPosition[0], listenerPosition[1]);
	}
	//Input excitation value at the defined excitation position//
	void inputExcitation(base_type_ excitation)	//For now, add excitation each time this is called - Will want to buffer it and pass into kernel.
	{
		nGrid()->valueAt(excitationPosition[0], excitationPosition[1]) += excitation;
	}

	//Print grid for debugging//
	void printGrid()
	{
		for (int y = 0; y != width_; y++)
		{
			for (int x = 0; x != height_; x++)
			{
				std::cout << nGrid()->valueAt(x, y) << " | ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
};

#endif