#include "../defines/Collisions.h"
#include "objectFactories.h"

#include "../../core/Rendering/viewProcess.h"
#include "../defines/renderingLayers.h"


#include "../ObjProcessors/GunProcessor.h"
#include "../ObjProcessors/OffsetProcessor.h"
#include "../ObjProcessors/GroundMoveProcessor.h"

#include "../../core/componentSys/processor/RenderProcessor.h"
#include "../../core/componentSys/processor/PhyProcessor.h"

#include "../../core/Rendering/renderUtil.h"


using namespace ObjectFactories;


/*
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
*/

Object *ObjectFactories::CreateGun(GunFactoryInfo &info){
	OffsetData offset;
	
	assert(info.parent != NULL);

	Object *gun = new Object("gun");
	info.parent->addChild(gun);

	vector2 *pos = gun->getPrimitive<vector2>(Hash::getHash("position"));
	*pos = info.pos;


	//renderer------------------------------------
	vector2 gunDim = vector2(2,1) * info.viewProc->getGame2RenderScale();


	sf::Shape *SFMLShape  = renderUtil::createRectangleShape(gunDim);
	SFMLShape->setFillColor(sf::Color::Red);
	SFMLShape->setFillColor(sf::Color(rand()% 256, rand()% 256, rand()% 256));
	SFMLShape->setOutlineColor(sf::Color::White);
	SFMLShape->setOutlineThickness(-2.0);

	RenderNode renderNode(SFMLShape, renderingLayers::action);
	RenderData renderData = RenderProcessor::createRenderData(&renderNode);

	//offset-------------------------------------
	offset.offsetAngle = false;
	

	//final---------------------------------
	gun->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(renderData));
	gun->addProp(Hash::getHash("GunData"), 
		new Prop<GunData>(info.gunData));
	
	
	gun->addProp(Hash::getHash("OffsetData"), 
		new Prop<OffsetData>(offset));
	
	return gun;	
};

Object *ObjectFactories::CreateBullet(BulletFactoryInfo &info){
	
	Object *bullet = new Object("bullet");
	vector2 *pos = bullet->getPrimitive<vector2>(Hash::getHash("position"));
	*pos = info.pos;

	//physics------------------------------------------------------------
	b2BodyDef bodyDef;
	bodyDef.type = b2_dynamicBody;

	b2CircleShape bulletBoundingBox;
	bulletBoundingBox.m_radius = info.radius;


	b2Filter collisionFilter;
	collisionFilter.categoryBits = CollisionGroups::BULLET;

	b2FixtureDef fixtureDef;
	fixtureDef.shape = &bulletBoundingBox;
	fixtureDef.friction = 0.0;
	fixtureDef.restitution = 0.0;
	fixtureDef.filter = collisionFilter;

	PhyData phy = PhyProcessor::createPhyData(&bodyDef, &fixtureDef);
	phy.collisionType = Hash::getHash("bullet");
	
	//renderer------------------------------------
	sf::Shape *SFMLShape  = renderUtil::createShape(&bulletBoundingBox, info.viewProc);
	SFMLShape->setFillColor(sf::Color::Red);

	RenderNode renderNode(SFMLShape, renderingLayers::action);
	RenderData renderData = RenderProcessor::createRenderData(&renderNode);

	//final---------------------------------
	bullet->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(renderData));
	bullet->addProp(Hash::getHash("PhyData"), 
		new Prop<PhyData>(phy));
	bullet->addProp(Hash::getHash("BulletData"), 
		new Prop<BulletData>(info.bulletData));
			
	return bullet;
};


Object *ObjectFactories::CreateEnemy(EnemyFactoryInfo &info) {

	Object *enemy = new Object("enemy");

	*enemy->getPrimitive<vector2>("position") = info.pos;


	//physics------------------------------------------------------------
	b2BodyDef bodyDef;
	bodyDef.type = b2_dynamicBody;
	bodyDef.gravityScale = 0.0f;

	b2CircleShape enemyBoundingBox;
	enemyBoundingBox.m_radius = 1.0f;

	b2FixtureDef fixtureDef;
	fixtureDef.shape = &enemyBoundingBox;
	fixtureDef.friction = 0.0;
	fixtureDef.restitution = 0.0;

	//phy.fixtureDef.push_back(fixtureDef);
	PhyData phy = PhyProcessor::createPhyData(&bodyDef, &fixtureDef); 
	phy.collisionType = Hash::getHash("enemy");


	//SFMLShape--------------------------------------------------------------------

	sf::Shape *SFMLShape = renderUtil::createShape(&enemyBoundingBox, info.viewProc);
	SFMLShape->setFillColor(sf::Color::White);
	SFMLShape->setOutlineColor(sf::Color::Black);
	SFMLShape->setOutlineThickness(-3.0);

	RenderNode renderNode(SFMLShape, renderingLayers::action);
	RenderData renderData = RenderProcessor::createRenderData(&renderNode);


	//final---------------------------------
	enemy->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(renderData));
	enemy->addProp(Hash::getHash("PhyData"), 
		new Prop<PhyData>(phy));

	return enemy;
};



#include "../terrainGen/terrain.h"
#include "../terrainGen/terrainGenerator.h"
Object *CreateCell();
Object *ObjectFactories::CreateTerrain(TerrainFactoryInfo &info){
	
	std::vector<AABB> AABBs = genTerrainChunks(info.terrain);
	//HACK!-----------------------------------------

	IO::infoLog<<"AABB's size: "<<AABBs.size()<<"\n";
	assert(AABBs.size() < 2000);


	b2PolygonShape phyShapes [2000];
	b2FixtureDef terrainFixtures [2000];
	RenderNode renderNodes [2000];


	int numAABBs = 0;

	for(auto terrainChunk : AABBs) {
		vector2 center  = terrainChunk.getCenter();
		vector2 halfDim = terrainChunk.getHalfDim();

		PRINTVECTOR2(halfDim)


		center = vector2( center.x * info.blockDim.x, center.y * info.blockDim.y);
		halfDim = vector2(halfDim.x * info.blockDim.x, halfDim.y * info.blockDim.y);


		//physics--------------------------------
		b2PolygonShape &phyShape = phyShapes[numAABBs];
		phyShape.SetAsBox(halfDim.x, halfDim.y, center, 0);
		
		b2FixtureDef &terrainFixture = terrainFixtures[numAABBs];
		terrainFixture.shape = &phyShape;
		terrainFixture.friction = 1.0;

		{
			//renderer------------------------------------
			float game2RenderScale = info.viewProc->getGame2RenderScale();
			sf::Shape *SFMLShape  = renderUtil::createShape(&phyShape, info.viewProc);
			SFMLShape->setFillColor(sf::Color::Blue);

			renderNodes[numAABBs].setRenderer(SFMLShape, renderingLayers::HUD);
			//RenderNode renderNode(SFMLShape, renderingLayers::HUD);
		}

		numAABBs++;
	}

	b2BodyDef bodyDef;
	bodyDef.type = b2_staticBody;

	PhyData phy = PhyProcessor::createPhyData(&bodyDef, terrainFixtures, numAABBs);
	phy.collisionType = Hash::getHash("terrain");

	RenderData renderData = RenderProcessor::createRenderData(renderNodes, numAABBs);

	Object *obj = new Object("terrain");
	obj->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(renderData));
	obj->addProp(Hash::getHash("PhyData"), 
		new Prop<PhyData>(phy));

	return obj;

};


Object *ObjectFactories::CreatePickup(PickupFactoryInfo &info){
	Object *obj = new Object("pickup");

	vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));
	*pos = info.pos;

	//physics------------------------------------------------------------
	b2BodyDef bodyDef;
	bodyDef.type = b2_staticBody;

	b2PolygonShape pickupShape;
	pickupShape.m_radius = info.radius;

	b2Filter collisionFilter;
	collisionFilter.categoryBits = CollisionGroups::PICKUP;

	b2FixtureDef fixtureDef;
	fixtureDef.shape = &pickupShape;
	fixtureDef.friction = 0.0;
	fixtureDef.restitution = 0.0;
	fixtureDef.isSensor = true;
	fixtureDef.filter = collisionFilter;

	PhyData phy = PhyProcessor::createPhyData(&bodyDef, &fixtureDef);
	phy.collisionType = Hash::getHash("pickup");

	//renderer------------------------------------
	float game2RenderScale = info.viewProc->getGame2RenderScale();
	sf::Shape *SFMLShape  = new sf::CircleShape(info.radius * game2RenderScale, 4);
	SFMLShape->setFillColor(sf::Color::Red);

	RenderNode renderNode(SFMLShape, renderingLayers::action);
	RenderData renderData = RenderProcessor::createRenderData(&renderNode);

	//final---------------------------------
	obj->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(renderData));
	obj->addProp(Hash::getHash("PhyData"), 
		new Prop<PhyData>(phy));
	obj->addProp(Hash::getHash("PickupData"), 
		new Prop<PickupData>(info.pickup));
	
	return obj;
};



Object *ObjectFactories::CreatePlayer(PlayerFactoryInfo &info) {

	Object *obj = new Object("player");
	
	vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));
	*pos = info.pos;

	
	//physics------------------------------------------------------------
	b2BodyDef bodyDef;
	bodyDef.type = b2_dynamicBody;

	b2CircleShape playerBoundingBox;
	playerBoundingBox.m_radius = 1.0f;

	b2FixtureDef fixtureDef;
	fixtureDef.shape = &playerBoundingBox;
	fixtureDef.friction = 0.0;
	fixtureDef.restitution = 0.0;

	PhyData phy = PhyProcessor::createPhyData(&bodyDef, &fixtureDef);
	phy.collisionType = Hash::getHash("player");

	//renderNode---------------------------------------------------------------
	sf::Shape *SFMLShape = renderUtil::createShape(&playerBoundingBox, info.viewProc);
	SFMLShape->setFillColor(sf::Color::Green);

	RenderNode renderNode(SFMLShape, renderingLayers::action);
	RenderData renderData = RenderProcessor::createRenderData(&renderNode);
	//movement-----------------------------------------------------------
	groundMoveData moveData;
	moveData.xVel = 60;
	moveData.xAccel = 3;
	moveData.movementDamping = vector2(0.05, 0.0);
	moveData.jumpRange = info.viewProc->getRender2GameScale() * 256;
	moveData.jumpHeight = info.viewProc->getRender2GameScale() * 128;
	moveData.jumpSurfaceCollision = Hash::getHash("terrain");
	
	//camera---------------------------------------------------------------
	info.cameraData.enabled = true;


	//final creation--------------------------------------------------------
	obj->addProp(Hash::getHash("PhyData"), 
		new Prop<PhyData>(phy));

	obj->addProp(Hash::getHash("RenderData"), 
		new Prop<RenderData>(renderData));

	obj->addProp(Hash::getHash("groundMoveData"), 
		new Prop<groundMoveData>(moveData));

	obj->addProp(Hash::getHash("CameraData"), 
		new Prop<CameraData>(info.cameraData));


	return obj;
};