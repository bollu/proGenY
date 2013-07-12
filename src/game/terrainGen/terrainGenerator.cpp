#pragma once
#include "terrainGenerator.h"



void setLevelDim(vector2 dim){
	this->levelDim = dim;
};
void setSeed(unsigned int seed){
	this->seed = seed;
};

void reserveChunk(vector2 pos);

void Generate(){

};


const std::vector<Chunk> &getLevel(){
	return this->chunks;
};

Chunk::Chunk(){
	this->filled = false;
}

vector2 Chunk::_limitChunkCoord(vector2 rawChunkCoord){
	vector2 minCoord = vector2(0,0);

	return rawChunkCoord.clamp(minCoord, levelDim);
};



