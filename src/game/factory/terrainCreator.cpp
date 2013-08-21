#pragma once
#include "terrainCreator.h"
#include <algorithm>
#include "../defines/renderingLayers.h"
#include "../terrainGen/terrain.h"

terrainCreator::terrainCreator ( viewProcess *viewProc ){
	terrainGen::Terrain t(vector2(10, 10));
	
	this->viewProc = viewProc;
}


void terrainCreator::setBounds ( vector2 bottomLeft, vector2 topRight, vector2 chunkDim ){
	this->bottomLeft = bottomLeft;
	this->topRight	 = topRight;

	vector2 delta = topRight - bottomLeft;

	this->numChunks.x     = std::ceil( delta.x / chunkDim.x );
	this->numChunks.y     = std::ceil( delta.y / chunkDim.y );
	this->totalChunkCount = this->numChunks.x * this->numChunks.y;
	//this->terrainGen.setDim( this->numChunks );
} //terrainCreator::setBounds


#include "objectFactory.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../../core/ObjProcessors/renderProcessor.h"
#include "../../core/Process/viewProcess.h"
#include "../../core/renderUtil.h"
#include "../../util/mathUtil.h"

Object *terrainCreator::createObject (){
	this->_genTerrainData();

	Object *terrain = new Object( "terrain" );

	//terrain->setProp< vector2 > ( Hash::getHash( "position" ), vector2( 0, 0 ) );

	/*
	   phyProp phy;
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

	                if(!c.isFilled()){
	                        continue;
	                }


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

	                shapeRenderNode *renderer = new shapeRenderNode(SFMLShape,
	                   renderingLayers::terrain);
	                render.addRenderer(renderer);



	        }

	   }



	   terrain->addProp(Hash::getHash("phyProp"),
	        new Prop<phyProp>(phy));

	   terrain->addProp(Hash::getHash("renderData"),
	        new Prop<renderData>(render));

	 */
	return (terrain);
} //terrainCreator::createObject


void terrainCreator::reserveRectSpace ( vector2 center, vector2 halfDim ){}

void terrainCreator::_genTerrainData (){}