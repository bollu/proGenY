#pragma once
#include "terrainCreator.h"
#include <algorithm>

terrainCreator::terrainCreator(viewProcess *viewProc){
	this->viewProc = viewProc;
};

void terrainCreator::setBounds(vector2 bottomLeft, vector2 topRight, 
	vector2 chunkDim){
	this->bottomLeft = bottomLeft;
	this->topRight = topRight;

	
	vector2 delta = topRight - bottomLeft;

	this->numChunks.x = std::ceil(delta.x / chunkDim.x);
	this->numChunks.y = std::ceil(delta.y / chunkDim.y);

	this->totalChunkCount = this->numChunks.x * this->numChunks.y;

	for(int i = 0; i < totalChunkCount; i++){
		this->chunks.push_back(Chunk());
	}

};

void terrainCreator::getEmptySpace(){


};



#include "objectFactory.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../../core/ObjProcessors/renderProcessor.h"
#include "../../core/Process/viewProcess.h"
#include "../../core/renderUtil.h"
#include "../../util/mathUtil.h"

Object *terrainCreator::createObject(){
	this->_genTerrainData();

	Object *terrain = new Object("terrain");
	terrain->setProp<vector2>(Hash::getHash("position"), vector2(0, 0));
	
	phyData phy;
	renderData render;
	render.centered = false;

	phy.collisionType = Hash::getHash("terrain");
	phy.bodyDef.type = b2_staticBody;


	vector2 posDelta = (this->topRight - this->bottomLeft);
	posDelta.x *= (1.0 / this->numChunks.x);
	posDelta.y *= (1.0 / this->numChunks.y);


	PRINTVECTOR2(posDelta);

	for(int y = 0; y < this->numChunks.y; y++){
		for(int x = 0; x < this->numChunks.x; x++){

			Chunk &c = this->_getChunk(vector2(x, y));

			if(!c.filled){
				continue;
			}

			vector2 gamePos = this->bottomLeft + vector2(posDelta.x * x
													, posDelta.y * y);

			b2PolygonShape *boundingBox = new b2PolygonShape();

			boundingBox->SetAsBox(posDelta.x / 2.0,
								posDelta.y / 2.0,
								vector2(gamePos.x, gamePos.y),
								0);
								
			b2FixtureDef fixtureDef;
			fixtureDef.shape = boundingBox;
			fixtureDef.friction = 0.0;
			fixtureDef.restitution = 0.0;

			phy.fixtureDef.push_back(fixtureDef);

		
			sf::Shape *SFMLShape = renderUtil::createShape(boundingBox, 
				viewProc);
			SFMLShape->setFillColor(sf::Color::Red);
	
			Renderer renderer(SFMLShape);
			render.addRenderer(renderer);

			

		}

	}
	
	
	
	terrain->addProp(Hash::getHash("phyData"), 
		new Prop<phyData>(phy));

	terrain->addProp(Hash::getHash("renderData"), 
		new Prop<renderData>(render));


	return terrain;
	
}



vector2 terrainCreator::_game2QuantizedCoord(vector2 realCoord){};

vector2 terrainCreator::_quantized2WorldCoord(vector2 quantizedCoord){};

vector2 terrainCreator::_limitQuantizedCoord(vector2 rawQuantizedCoord){
	vector2 limitedCoord = rawQuantizedCoord;

	return limitedCoord.clamp(vector2(0,0), vector2(numChunks.x - 1, numChunks.y - 1));
};


//position in terrain's coordinate system
terrainCreator::Chunk& terrainCreator::_getChunk(vector2 quantizedPos){

	
	assert(quantizedPos.x < this->numChunks.x &&
		quantizedPos.y < this->numChunks.y);

	return this->chunks[quantizedPos.x + (this->numChunks.x * quantizedPos.y)];
};

//flood fills chunks in the bounding rectangle.center of rectangle
//is quantizecdCenter. Overwrites values. 
int terrainCreator::_floodFillChunks(vector2 quantizedCenter,
 vector2 halfDim, chunkType type){

	vector2 maxPos = this->_limitQuantizedCoord(quantizedCenter + halfDim);
	vector2 minPos =  this->_limitQuantizedCoord(quantizedCenter - halfDim);


	vector2 currentPos = vector2(maxPos.x, maxPos.y);
	int chunksAffected = 0;

	while(currentPos.y >= minPos.y){
		while(currentPos.x >= minPos.x){

			this->_getChunk(currentPos).filled = type;
			currentPos.x--;
			chunksAffected++;
		}
		currentPos.x = maxPos.x;
		currentPos.y--;
	}

	return chunksAffected;
};

int terrainCreator::_negateChunks(vector2 quantizedCenter, vector2 halfDim){
	vector2 maxPos = this->_limitQuantizedCoord(quantizedCenter + halfDim);
	vector2 minPos =  this->_limitQuantizedCoord(quantizedCenter - halfDim);


	vector2 currentPos = vector2(maxPos.x, maxPos.y);
	int chunksAffected = 0;


	while(currentPos.y >= minPos.y){
		while(currentPos.x >= minPos.x){

			terrainCreator::Chunk &chunk = this->_getChunk(currentPos);
			chunk.filled = !chunk.filled;

			currentPos.x--;
			chunksAffected++;
		}
		currentPos.x = maxPos.x;
		currentPos.y--;
	}

	return chunksAffected;
};

float terrainCreator::_getDensity(){
	int totalActive = 0;

	vector2 currentPos = vector2(numChunks.x - 1, numChunks.y - 1);

	while(currentPos.y >= 0){
		while(currentPos.x >= 0){

			
			terrainCreator::Chunk &chunk = this->_getChunk(currentPos);
			
			if(chunk.filled){
				totalActive ++; 

			};
			currentPos.x--;

		}

		currentPos.x = numChunks.x - 1;
		
		currentPos.y--;
	}

	return (float)totalActive / this->totalChunkCount;
};

void terrainCreator::_genTerrainData(){
	vector2 currentPos = vector2(std::floor(this->numChunks.x / 2.0),
								std::floor(this->numChunks.y / 2.0));
	vector2 halfDim = vector2(5,5);

	for(int i = 0; i < 10; i++){


		int prob = rand() % 30;
		if(prob < 16){
			this->_floodFillChunks(currentPos, halfDim, true);
		}
		else{
			this->_floodFillChunks(currentPos, halfDim, false);
		}

		halfDim.x = rand() % (int)this->numChunks.x;
		halfDim.y = rand() % (int)this->numChunks.y;

		halfDim *= 0.5;

		halfDim.x = std::ceil(halfDim.x);
		halfDim.y = std::ceil(halfDim.y);

		currentPos.x = rand() % (int)this->numChunks.x;
		currentPos.y = rand() % (int)this->numChunks.y;

	}


	
	halfDim.x = 2.0 * std::ceil(std::sqrt(this->numChunks.x)); 
	halfDim.y = 2.0 * std::ceil(std::sqrt(this->numChunks.x));
	for(int i = 0; i < 3; i++){
		currentPos.x = rand() % (int)this->numChunks.x;
		currentPos.y = rand() % (int)this->numChunks.y;

		this->_negateChunks(currentPos, halfDim);
	}



	halfDim.x = std::ceil(std::sqrt(this->numChunks.x) / 4.0); 
	halfDim.y = std::ceil(std::sqrt(this->numChunks.x) / 4.0);

	float density = this->_getDensity();
	
	for(int i = 0; i < 1000; i++){


		currentPos.x = rand() % (int)this->numChunks.x;
		currentPos.y = rand() % (int)this->numChunks.y;


		int chunksAffected = 0;
		
		if(density > 0.5){
			chunksAffected = this->_floodFillChunks(currentPos, halfDim, false);
			density -= (float)chunksAffected / this->totalChunkCount;
		}else{
			chunksAffected = this->_floodFillChunks(currentPos, halfDim, true);
			density += (float)chunksAffected / this->totalChunkCount;
		}
	}


	//this->_dbgPrintChunks();
};


#include <fstream>
void terrainCreator::_dbgPrintChunks(){
	std::ofstream mapFile("mapDump");

	for(int y = this->numChunks.y - 1; y >= 0; y--){
		for(int x = 0; x < this->numChunks.x - 1; x++){

			terrainCreator::Chunk &c = this->_getChunk(vector2(x, y));

			if(c.filled){
				mapFile<<"X";
			}
			else{
				mapFile<<".";
			}
		}
		mapFile<<"\n";
	}

	mapFile.close();
};
