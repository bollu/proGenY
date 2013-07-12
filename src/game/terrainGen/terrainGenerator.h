#pragma once
#include <vector>




class terrainGenerator{
public:
	struct Chunk{
		bool filled;
		Chunk();

		const bool isFilled(){
			return this->filled;
		}
	};

	

public:
	void setLevelDim(vector2 dim);
	void setSeed(unsigned int seed);

	void reserveChunk(vector2 pos);

	void Generate();


	const std::vector<Chunk> &getLevel();

private:
	unsigned int seed;
	vector2 levelDim;
	float numTiles;

	std::vector<Chunk> chunks;

	vector2 _limitChunkCoord(vector2 rawChunkCoord);
};