#pragma once
#include <vector>
#include "../../core/vector.h"

struct phyData;
struct renderData;

class terrainGenerator{
public:

	enum chunkType{
		empty = 0,
		filled = 1,
		triBottomLeft,
		triBottomRight,
		triTopLeft,
		triTopRight,
	};

	struct Chunk{

		chunkType type;
		Chunk(chunkType _type = empty) : type(_type){

		};

		const bool isFilled(){
			return this->type != chunkType::empty;
		}

		const chunkType getType(){
			return this->type;
		}


	};

	

public:
	void setDim(vector2 levelDim);
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

	void _addChunkToObj(Chunk &c, phyData &phy, renderData &render);
};