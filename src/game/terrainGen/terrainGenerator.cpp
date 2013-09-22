#pragma once
#include "terrainGenerator.h"

#define BlockV(v) this->blocks[(v).x * numBlocks.x + (v).y]
#define BlockI(xPos, yPos) this->blocks[((xPos) * (numBlocks.x)) + (yPos)]

terrainGenerator::terrainGenerator(unsigned int seed, vector2 numBlocks) : noiseGen(seed){
	this->seed = seed;
	this->numBlocks = vector2(floor(numBlocks.x), floor(numBlocks.y));
	

	this->_InitBlocks();

	this->_genRoom(AABB(vector2(2, 2),vector2(5, 5)));

	this->_genTerrain(vector2(1, 1), vector2(5, 5));

};

void terrainGenerator::_InitBlocks(){
	int totalArea = this->numBlocks.x * this->numBlocks.y;
	
	this->blocks.reserve(totalArea);

	/*for(terrainGenerator::blockType &block : blocks){
		block = blockType::filled;
	}*/


};

bool terrainGenerator::_isRoomLegal(const AABB &roomToCheck){
	for(const AABB &room : rooms){
		if(room.Intersects(roomToCheck)){
			return false;
		}
	}

	return true;
};

void terrainGenerator::_genRoom(const AABB &room){

	vector2 center = room.getCenter();
	vector2 halfDim = room.getHalfDim();

	int yTop = center.y + halfDim.y;
	int yBottom =  center.y - halfDim.y;

	int xRight = center.x + halfDim.x;
	int xLeft =  center.x - halfDim.x;

	/*
	for(int y = yBottom; y <= yTop; y++){
		BlockI(xLeft, y) = blockType::filled;
		BlockI(xRight, y) = blockType::filled;
	}

	for(int x = xLeft + 1; x <= xRight - 1; x++){
		BlockI(x, yTop) = blockType::filled;
		BlockI(x, yBottom) = blockType::filled;

		for(int y = yBottom + 1; y <= yTop - 1; y++ ){
			BlockI(x,y) = blockType::insideRoom;
		}
	}*/
	for(int x = xLeft; x <= xRight; x++){
		for(int y = yBottom; y <= yTop; y++){
			BlockI(x, y) = blockType::empty;
		}
	}

	/*for(int y = yBottom; y <= yTop; y++){
		BlockI(xLeft, y) = blockType::filled;
		BlockI(xRight, y) = blockType::filled;
	}*/
};

void terrainGenerator::_genCorridor(const AABB &currentRoom, const AABB &prevRoom){

};


terrainGenerator::blockType terrainGenerator::getBlockType(vector2 pos){

	/*oat normalizedX = pos.x / numBlocks.x;
	float normalizedY = pos.x / numBlocks.y;

	normalizedX *= 10;
	normalizedY *= 10;

	normalizedX += rand() % 2 == 0 ? -1 : 1 * (rand() % 100) / 9900.0;
	normalizedY += rand() % 2 == 0 ? -1 : 1 * (rand() % 100) / 9900.0;

	double density = this->noiseGen.noise(normalizedX, normalizedY,
	 seed);


	assert(density >= -1 && density <= 1);

	return _density2BlockType(pos, density);*/

	/*if(BlockV(pos) == blockType::insideRoom){
		return blockType::filled;
	};

	return blockType::empty;*/

	
	//return BlockV(pos);
	return blockType::empty;



};

terrainGenerator::blockType terrainGenerator::_density2BlockType(vector2 pos, double density){

	if(density < 0.5){
		return blockType::empty;
	}

	return blockType::filled;	
};

void terrainGenerator::_genTerrain(vector2 minHalfDim, vector2 maxHalfDim){

	
	bool done = false;

	int numIterations = 0;
	int numRooms = 0;

	while(!done && (numIterations < 1000) ){


		vector2 halfDim = util::lerp(minHalfDim, maxHalfDim, util::randFloat()); 
		vector2 center = util::lerp(nullVector, numBlocks, util::randFloat()); 

		AABB room(halfDim, center);
		numIterations++;

		if(!_isRoomLegal(room)){
			continue;
		}
		//okay, the room is legal

		this->_genRoom(room);
		this->rooms.push_back(room);

		numRooms++;
		
	}

};
