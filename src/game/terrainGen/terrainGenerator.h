#pragma once
#include "./../core/Object.h"
#include "./../core/AABB.h"
#include "../generators/PerlinNoise.h"





class terrainGenerator{
public:

	enum blockType{
	/*so that blocks are filled by default*/
	filled = 0,
	empty,
	

	numBlockTypes
	};

	terrainGenerator(unsigned int seed, vector2 numBlocks);

	blockType getBlockType(vector2 pos);

private:
	vector2 numBlocks;
	unsigned int seed;
	PerlinNoise noiseGen;

	std::vector<blockType> blocks;

	std::vector<AABB> rooms;
	AABB *prevRoom;
	
	void _InitBlocks();

	bool _isRoomLegal(const AABB &room);
	void _genRoom(const AABB &room);
	void _genCorridor(const AABB &currentRoom, const AABB &prevRoom);


	void _genTerrain(vector2 minHalfDim, vector2 maxHalfDim);

	blockType _density2BlockType(vector2 pos, double density);
};
