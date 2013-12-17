#pragma once
#include "../controlFlow/Process.h"
#include "../controlFlow/processMgr.h"
#include "../include/SFML/Graphics.hpp"
#include "../controlFlow/eventMgr.h"
#include "../math/mathUtil.h"
#include "windowProcess.h"

#include <list>
#include <thread>
#include <atomic>
#include  <mutex>


/*!
 follows openGL coordinate system. z-axis is down the screen. so, (0, 0, -100) will be 
 more in the back that (0, 0, -90)
 */
class renderProcess: public Process{
public:
	
	class baseRenderNode{
		public:
		int z;
		bool drawDisabled;

		baseRenderNode(){
			this->drawDisabled = false;;
			this->z = 0;
		}

		virtual void Draw(sf::RenderWindow *window) = 0;
		virtual void setPosition(vector2 position) = 0;
		virtual void setRotation(util::Angle angle) = 0;	
	};


	template <typename T>
	class renderNode : public baseRenderNode{
		T *drawable;
		
		public:
		renderNode(T *_drawable) : drawable(_drawable){
			assert(drawable != NULL);
			this->z = 0;
		};

		renderNode(T *_drawable, int z) : drawable(_drawable){
			assert(drawable != NULL);
			this->z = z;
		};

		void Draw(sf::RenderWindow *window){

			if(!this->drawDisabled){
				window->draw(*this->drawable);
			}
		}

		void setPosition(vector2 position){
			this->drawable->setPosition(position);
		};

		void setRotation(util::Angle angle){
			this->drawable->setRotation(angle.toDeg());
		};	

	};

	renderProcess(processMgr &processManager, Settings &settings, eventMgr &_eventManager);
	void Draw();

	void addRenderNode(renderProcess::baseRenderNode *node);
	void removeRenderNode(renderProcess::baseRenderNode *node);

private:
	sf::RenderWindow *window;
	std::list<baseRenderNode *> nodes;

	/*!return true if first argument goes *before the 2nd*
	we want things at the back to be drawn first, so return 
	true if the first object is behind the 2nd object 
	i.e - if the Z of the first object is less that the 2nd object
	(as -100 < -10)
	*/ 
	

	static bool sortFn(baseRenderNode *first, baseRenderNode *second){
		if(first->z < second->z){
			return true;
		}

		return false;
	}	
};

typedef renderProcess::renderNode<sf::Shape> shapeRenderNode;
typedef renderProcess::renderNode<sf::Shape> spriteRenderNode;
typedef renderProcess::renderNode<sf::Shape> textRenderNode;
