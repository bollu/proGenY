#pragma once


enum terrainType {
	Empty = 0,
	Filled
};

struct Terrain {
private:	
	unsigned int width_, height_;
	terrainType *terrain;

	unsigned int maxHeight_;

public:

	Terrain(unsigned int width, unsigned int height) : width_(width), height_(height), maxHeight_(0) {
		terrain = new terrainType[width * height];

		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				terrain[x + width_ * y] = terrainType::Empty;
			}
		}
	}

	terrainType At(int x, int y) {
		assert(x < width_ && y < height_);
		return terrain[x + width_ * y];
	}

	void Set(int x, int y, terrainType type) {
		assert(x < width_ && y < height_);
		terrain[x + width_ * y] = type;
	}

	unsigned int getWidth () {
		return width_;
	}

	unsigned int getHeight () {
		return height_;
	}

	unsigned int getHeightAt(int x) {
		//check from the top to the bottom
		for(int y = height_ - 1; y >= 0; y--) {
			if (terrain[x + width_ * y] == terrainType::Filled) {
				return y;
			}
		}

		return 0;
	}

	unsigned int getMaxHeight () {
		return maxHeight_;
	}

	void setMaxHeight (unsigned int maxHeight) {
		maxHeight_ = maxHeight;
	}
};