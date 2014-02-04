#pragma once
#include "../controlFlow/Process.h"
#include "../controlFlow/processMgr.h"
#include "../include/SFML/Graphics.hpp"
#include "../controlFlow/EventManager.h"
#include "../math/mathUtil.h"
#include "windowProcess.h"

#include "ParticleSystem.h"

#include <list>
#include <thread>
#include <atomic>
#include  <mutex>


struct RenderNode{
	int z_;
	bool disabled_;

	enum Type{
		Sprite,
		Text,
		Shape,
		Trail,
		Particle,
		None,
	} type_;

	union {
		sf::Shape *shape;
		sf::Text *text;
		sf::Sprite *sprite;
		ParticleSystem *particleSystem;
	} renderer_;

	
	static RenderNode ParticleSystemNode(ParticleSystem *particleSystem, int z) {
		RenderNode node;
		node.setParticleSystem(particleSystem, z);

		return node;
	}

	static RenderNode ShapeNode(sf::Shape *shape, int z) {
		RenderNode node;
		node.setShape(shape, z);

		return node;
	}

	RenderNode() {
		z_ = 0;
		disabled_ = true;
		type_ = Type::None;
	}

	void setShape(sf::Shape *shape, int z) {
		this->renderer_.shape = shape;
		type_ = Type::Shape;
		z_ = z;
		disabled_ = false;
	}

	void setParticleSystem(ParticleSystem *particleSystem, int z) {
		renderer_.particleSystem = particleSystem;
		type_ = Type::Particle;
		z_ = z;
		disabled_ = false;
	}


};

void setRenderNodePosition(RenderNode &renderNode, vector2 position);
void setRenderNodeAngle(RenderNode &renderNode, util::Angle angle);
void freeRenderNode(RenderNode &renderNode);
/*!
 follows openGL coordinate system. z-axis is down the screen. so, (0, 0, -100) will be 
 more in the back that (0, 0, -90)
 */
class renderProcess: public Process{
public:

	renderProcess(processMgr &processManager, Settings &settings, EventManager &_eventManager);
	void Update(float dt);
	void Draw();

	void addRenderNode(RenderNode &node);
	void removeRenderNode(RenderNode &node);

private:
	sf::RenderWindow *window;
	std::list<RenderNode *> nodes;

	float dt_;

	/*!return true if first argument goes *before the 2nd*
	we want things at the back to be drawn first, so return 
	true if the first object is behind the 2nd object 
	i.e - if the Z of the first object is less that the 2nd object
	(as -100 < -10)
	*/ 
	static bool sortFn(RenderNode *first, RenderNode *second){
		if(first->z_ < second->z_){
			return true;
		}

		return false;
	}

	void drawSprite_(RenderNode &node);
	void drawShape_(RenderNode &node);
	void drawText_(RenderNode &node);
	void drawParticleSystem_(RenderNode &node);
};
