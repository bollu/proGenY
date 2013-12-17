#include "objectFactories.h"

#include "../../core/Rendering/viewProcess.h"
#include "../defines/renderingLayers.h"


#include "../ObjProcessors/GunProcessor.h"
#include "../ObjProcessors/OffsetProcessor.h"

#include "../../core/componentSys/processor/RenderProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"
#include "../../core/Rendering/renderUtil.h"


using namespace ObjectFactories;


Object *ObjectFactories::CreateBoundary(BoundaryFactoryInfo &info){

	Object *boundaryObject = new Object("boundary");

	vector2 *pos = boundaryObject->getPrimitive<vector2>(
		Hash::getHash("position"));
	*pos = vector2(0,0);

	
	PhyData physicsData;
	RenderData render;

	physicsData.bodyDef.type = b2_staticBody;
	physicsData.collisionType = Hash::getHash("terrain");

	{
		//BOTTOM---------------------------------------------------------
		b2PolygonShape *bottom = new b2PolygonShape(); 
		vector2 bottomCenter = vector2(info.levelDim.x / 2.0, 0);//vector2(info.levelDim.x / 2.0, 0);
		bottom->SetAsBox(info.levelDim.x / 2.0, info.thickness, bottomCenter, 0);

		b2FixtureDef bottomFixtureDef;
		bottomFixtureDef.shape = bottom;
		bottomFixtureDef.friction = 1.0;

		physicsData.fixtureDef.push_back(bottomFixtureDef);

		sf::Shape *shape = renderUtil::createShape(bottom, 
			info.viewProc);
		shape->setFillColor(sf::Color::Blue);
		
		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::terrain);
		render.addRenderer(renderer);

	}

	{
		//TOP---------------------------------------------------------
		b2PolygonShape *top = new b2PolygonShape(); 
		vector2 topCenter = vector2(info.levelDim.x / 2.0, info.levelDim.y);//vector2(info.levelDim.x / 2.0, info.levelDim.y / 2.0);
		top->SetAsBox(info.levelDim.x / 2.0, info.thickness, topCenter, 0);

		b2FixtureDef topFixtureDef;
		topFixtureDef.shape = top;
		topFixtureDef.friction = 1.0;
		topFixtureDef.restitution = 0.0;

		physicsData.fixtureDef.push_back(topFixtureDef);


		sf::Shape *shape = renderUtil::createShape(top, 
			info.viewProc);
		shape->setFillColor(sf::Color::Blue);
		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::terrain);
		render.addRenderer(renderer);

	}

	
	{
		//LEFT---------------------------------------------------------
		b2PolygonShape *left = new b2PolygonShape(); 
		vector2 leftCenter = vector2(0, info.levelDim.y / 2.0);//vector2(0, info.levelDim.y / 2.0);
		left->SetAsBox(info.thickness, info.levelDim.y / 2.0, leftCenter, 0);

		b2FixtureDef leftFixtureDef;
		leftFixtureDef.shape = left;
		leftFixtureDef.friction = 1.0;

		physicsData.fixtureDef.push_back(leftFixtureDef);

		sf::Shape *shape = renderUtil::createShape(left, 
			info.viewProc);
		shape->setFillColor(sf::Color::Blue);
		
		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::terrain);
		render.addRenderer(renderer);
	}

	{
		//RIGHT---------------------------------------------------------
		b2PolygonShape *right = new b2PolygonShape(); 
		vector2 rightCenter = vector2(info.levelDim.x, info.levelDim.y / 2.0);
		right->SetAsBox(info.thickness, info.levelDim.y / 2.0, rightCenter, 0);

		b2FixtureDef rightFixtureDef;
		rightFixtureDef.shape = right;
		rightFixtureDef.friction = 1.0;

		physicsData.fixtureDef.push_back(rightFixtureDef);

		sf::Shape *shape = renderUtil::createShape(right, 
			info.viewProc);
		shape->setFillColor(sf::Color::Blue);
		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::terrain);
		render.addRenderer(renderer);
	}

	boundaryObject->addProp(Hash::getHash("PhyData"), 
		new Prop<PhyData>(physicsData));
	boundaryObject->addProp(Hash::getHash("RenderData"),
	 new Prop<RenderData>(render));

	return boundaryObject;
};

Object *ObjectFactories::CreateGun(GunFactoryInfo &info){
	RenderData render;
	OffsetData offset;
	
	assert(info.parent != NULL);

	Object *gun = new Object("gun");
	info.parent->addChild(gun);

	vector2 *pos = gun->getPrimitive<vector2>(Hash::getHash("position"));
	*pos = info.pos;


	//renderer------------------------------------
	vector2 gunDim = vector2(2,1) * info.viewProc->getGame2RenderScale();
	sf::Shape *shape = renderUtil::createRectangleShape(gunDim);
	shape->setFillColor(sf::Color::Blue);
	shape->setOutlineColor(sf::Color::White);
	shape->setOutlineThickness(-2.0);

	shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::aboveAction);
	render.addRenderer(renderer);
	
	//offset-------------------------------------
	offset.offsetAngle = false;
	

	//final---------------------------------
	gun->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(render));
	gun->addProp(Hash::getHash("GunData"), 
		new Prop<GunData>(info.gunData));
	
	
	gun->addProp(Hash::getHash("OffsetData"), 
		new Prop<OffsetData>(offset));
	
	return gun;	
};

Object *ObjectFactories::CreateBullet(BulletFactoryInfo &info){
	RenderData render;
	PhyData phy;
	
	
	Object *obj = new Object("bullet");
	vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));
	*pos = info.pos;

	//physics------------------------------------------------------------
	phy.collisionType = Hash::getHash("bullet");
	phy.bodyDef.type = b2_dynamicBody;
	//phy.bodyDef.type = b2_kinematicBody;
	phy.bodyDef.bullet = false;

	
	b2CircleShape *shape = new b2CircleShape();
	shape->m_radius = info.radius;

	b2FixtureDef fixtureDef;
	fixtureDef.shape = shape;
	fixtureDef.friction = 0.0;
	fixtureDef.restitution = 0.0;
	fixtureDef.isSensor = true;
	

	phy.fixtureDef.push_back(fixtureDef);


	//renderer------------------------------------
	sf::Shape *sfShape = renderUtil::createShape(shape, 
		info.viewProc);

	sfShape->setFillColor(sf::Color::Red);

	shapeRenderNode *shapeRenderer = new shapeRenderNode(sfShape);
	render.addRenderer(shapeRenderer);
	

	//final---------------------------------
	obj->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(render));
	obj->addProp(Hash::getHash("PhyData"), 
		new Prop<PhyData>(phy));
	obj->addProp(Hash::getHash("BulletData"), 
		new Prop<BulletData>(info.bulletData));
		
	IO::infoLog<<"bullet Created"<<IO::flush;
	
	return obj;
};




#include "../terrainGen/terrain.h"
#include "../terrainGen/terrainGenerator.h"
Object *CreateCell();
Object *ObjectFactories::CreateTerrain(TerrainFactoryInfo &info){
	PhyData phy;
	phy.bodyDef.type = b2_staticBody;
	phy.collisionType = Hash::getHash("terrain");

	RenderData render;

	std::vector<AABB> AABBs = genTerrainChunks(info.terrain);

	for(auto terrainChunk : AABBs) {
		vector2 center  = terrainChunk.getCenter();
		vector2 halfDim = terrainChunk.getHalfDim();

		PRINTVECTOR2(halfDim)


		center = vector2( center.x * info.blockDim.x, center.y * info.blockDim.y);
		halfDim = vector2(halfDim.x * info.blockDim.x, halfDim.y * info.blockDim.y);


		//physics--------------------------------
		b2PolygonShape *phyShape = new b2PolygonShape(); 
			
		phyShape->SetAsBox(halfDim.x, halfDim.y, center, 0);

		b2FixtureDef fixtureDef;
		fixtureDef.shape = phyShape;
		fixtureDef.friction = 1.0;

		phy.fixtureDef.push_back(fixtureDef);

		{
		//renderer------------------------------------
		sf::Shape *shape = renderUtil::createShape(phyShape, 
		info.viewProc);

		shape->setFillColor(sf::Color::Blue);

		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::HUD);
		render.addRenderer(renderer);
		}

	}


	Object *obj = new Object("terrain");
	obj->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(render));
	obj->addProp(Hash::getHash("PhyData"), 
		new Prop<PhyData>(phy));

	return obj;

};


Object *ObjectFactories::CreatePickup(PickupFactoryInfo &info){
	RenderData render;
	PhyData phy;
	
	Object *obj = new Object("pickup");

	vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));
	*pos = info.pos;

	//physics------------------------------------------------------------
	phy.collisionType = Hash::getHash("pickup");
	phy.bodyDef.type = b2_staticBody;

	
	b2CircleShape *shape = new b2CircleShape();
	shape->m_radius = info.radius;
	b2FixtureDef fixtureDef;
	fixtureDef.shape = shape;
	fixtureDef.friction = 0.0;
	fixtureDef.restitution = 0.0;
	fixtureDef.isSensor = true;

	phy.fixtureDef.push_back(fixtureDef);


	//renderer------------------------------------
	float game2RenderScale = info.viewProc->getGame2RenderScale();
	sf::Shape *sfShape = new sf::CircleShape(info.radius * game2RenderScale,
									4);
	

	sfShape->setFillColor(sf::Color::Red);

	shapeRenderNode* renderer = new shapeRenderNode(sfShape);
	render.addRenderer(renderer);
	

	//final---------------------------------
	obj->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(render));
	obj->addProp(Hash::getHash("PhyData"), 
		new Prop<PhyData>(phy));
	obj->addProp(Hash::getHash("PickupData"), 
		new Prop<PickupData>(info.pickup));
	
	return obj;
};