#include "objectFactories.h"

#include "../../core/Rendering/viewProcess.h"
#include "../defines/renderingLayers.h"


#include "../ObjProcessors/gunProcessor.h"
#include "../ObjProcessors/offsetProcessor.h"

#include "../../core/componentSys/processor/renderProcessor.h"
#include "../../core/componentSys/processor/phyProcessor.h"
#include "../../core/Rendering/renderUtil.h"


using namespace ObjectFactories;


Object *ObjectFactories::CreateBoundary(BoundaryFactoryInfo &info){

	Object *boundaryObject = new Object("boundary");

	vector2 *pos = boundaryObject->getPrimitive<vector2>(
		Hash::getHash("position"));
	*pos = vector2(0,0);

	
	phyData physicsData;
	renderData render;

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

	boundaryObject->addProp(Hash::getHash("phyData"), 
		new Prop<phyData>(physicsData));
	boundaryObject->addProp(Hash::getHash("RenderData"),
	 new Prop<renderData>(render));

	return boundaryObject;
};

Object *ObjectFactories::CreateGun(GunFactoryInfo &info){
	renderData render;
	offsetData offset;
	
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
	//render.centered = true;
	
	//offset-------------------------------------
	assert(info.parent != NULL);
	offset.parent = info.parent;
	offset.offsetAngle = false;
	

	//final---------------------------------
	gun->addProp(Hash::getHash("RenderData"), 
		new Prop<renderData>(render));
	gun->addProp(Hash::getHash("GunData"), 
		new Prop<gunData>(info.gunData));
	gun->addProp(Hash::getHash("OffsetData"), 
		new Prop<offsetData>(offset));
	
	return gun;	
};

Object *ObjectFactories::CreateBullet(BulletFactoryInfo &info){
	renderData render;
	phyData phy;
	
	
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
	obj->addProp(Hash::getHash("renderData"), 
		new Prop<renderData>(render));
	obj->addProp(Hash::getHash("phyData"), 
		new Prop<phyData>(phy));
	obj->addProp(Hash::getHash("BulletData"), 
		new Prop<bulletData>(info.bulletData));
	
	return obj;
};


#include "../terrainGen/terrain.h"
Object *CreateCell();
Object *ObjectFactories::CreateTerrain(TerrainFactoryInfo &info){
	phyData phy;
	phy.bodyDef.type = b2_staticBody;
	phy.collisionType = Hash::getHash("terrain");

	renderData render;

	

	unsigned int w = info.terrain.getWidth(), h = info.terrain.getHeight();

	for(int y = 0; y < h; y++) {
		for(int x = 0; x < w; x++) {
			

			auto cell = info.terrain.At(x, y);

			if (cell == terrainType::Empty) {
				continue;
			}
			vector2 center = vector2((x + 0.5) * info.blockDim.x, (y + 0.5) * info.blockDim.y);
			
			//physics--------------------------------
			b2PolygonShape *phyShape = new b2PolygonShape(); 
			
			phyShape->SetAsBox(info.blockDim.x / 2.0, info.blockDim.y / 2.0, center, 0);

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
	}

	Object *obj = new Object("terrain");
	obj->addProp(Hash::getHash("renderData"), 
		new Prop<renderData>(render));
	obj->addProp(Hash::getHash("phyData"), 
		new Prop<phyData>(phy));

	return obj;

};