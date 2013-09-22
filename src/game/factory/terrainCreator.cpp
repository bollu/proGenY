#pragma once
#include "terrainCreator.h"
#include "../terrainGen/terrainGenerator.h"


Object *terrainCreator::createObject(vector2 pos) const{

	phyData phy;
	
	Object *obj = new Object("terrain");

	terrainGenerator generator(seed, numBlocks);

	bool blockCreated = false;

	for(int x = 0; x < numBlocks.x; x++){
		for(int y = 0; y < numBlocks.x; y++){

			terrainGenerator::blockType type = generator.getBlockType(vector2(x, y));

			if(type == terrainGenerator::blockType::filled){
				_createFilledBlock(phy, vector2(x, y));
				blockCreated = true;
			}
		}
	}

	if(blockCreated){
		renderData render = _createRenderer(phy);
		obj->addProp(Hash::getHash("phyData"), 
			new Prop<phyData>(phy));
		obj->addProp(Hash::getHash("renderData"),
			new Prop<renderData>(render));
	}
	return obj;
};

terrainCreator::terrainCreator(viewProcess *_viewProc) : viewProc(_viewProc){};

void terrainCreator::Init(vector2 numBlocks, vector2 blockDim, unsigned int seed){


	//this->numBlocks = numBlocks;
	this->numBlocks = vector2(floor(numBlocks.x), floor(numBlocks.y));
	PRINTVECTOR2(numBlocks);
	
	this->blockDim = blockDim;
	this->seed = seed;
}


void terrainCreator::_createFilledBlock(phyData &phy, vector2 blockPos) const{
	b2PolygonShape *box = new b2PolygonShape(); 
	box->SetAsBox(blockDim.x * 0.5, blockDim.y * 0.5, _block2World(blockPos),0);

	b2FixtureDef boxFixtureDef;
	boxFixtureDef.shape = box;
	boxFixtureDef.friction = 1.0;
	boxFixtureDef.isSensor = true;

	phy.fixtureDef.push_back(boxFixtureDef);
}


renderData terrainCreator::_createRenderer(phyData &phy) const{

	renderData render;

	for(b2FixtureDef &fixtureDef : phy.fixtureDef){

		sf::Shape *shape = renderUtil::createShape(fixtureDef.shape, viewProc);
		shape->setFillColor(sf::Color::Black);
			
		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::BACK);
		render.addRenderer(renderer);
	}

	return render;
};


vector2 terrainCreator::_block2World(vector2 blockPos) const{
	vector2 base = -1 * blockDim * 0.5;
	vector2 delta = vector2(blockDim.x * blockPos.x, blockDim.y * blockPos.y);

	return base + delta;
};
