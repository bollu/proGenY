#pragma once
#include "../objectProcessor.h"
#include "../include/SFML/Graphics.hpp"
#include "../vector.h"
#include <vector>
#include "../../util/strHelper.h"

#include "../Process/processMgr.h"
#include "../Settings.h"
#include "../Messaging/eventMgr.h"
#include "../Process/viewProcess.h"

/*!Represents an entity that can be rendered
Anything in the Engine that has to be rendered has to be one of these
types: a Sprite, a Shape, or Text. This class encapsulates all of the
types that are used to Render.

\sa  renderData
*/
class Renderer{
public:
	enum Type{
		Shape = 0,
		Sprite,
		Text,

	};

	/*!Construct a Renderer with a Shape */
	Renderer(sf::Shape *shape);
	/*!Construct a Renderer with a Sprite */
	Renderer(sf::Sprite *sprite);
	/*! Construct a Renderer with a Text */
	Renderer(sf::Text *text);



	Renderer(const Renderer &other);

private:
	friend class renderProcessor;
	friend class renderData;

	union{
		sf::Shape *shape;
		sf::Sprite *sprite;
		sf::Text *text;
	 } data;

	 Renderer::Type type;

	 /*! Deletes the pointer that it owns. called by renderData
	This can't be done in the destructor since the Renderer
is supposed to be created on the stack, and then passed
	on to renderData.  */
	void deleteData();
};


/*!The data that is used by the renderProcessor to render objects
It's a collection of Renderer objects. as one Object may like to render
multiple things, the renderData class acts as a "bag" to hold multiple Renderer
objects

\sa renderProcesor Renderer
*/
class renderData{
private:
	friend class renderProcessor;
	std::vector<Renderer>renderers;
public:
	/*! Add a Renderer the renderData */
	void addRenderer(Renderer &renderer){
		this->renderers.push_back(renderer);
	}

	//whether the renderer should be shifted by half.
	//useful when manually setting positions / screwing around
	bool centered;

	renderData() : centered(false){};
};

/*! an objectProcessor that handles rendering Object
the Object must be attached with renderData. the objectProcessor
uses renderData to draw the Object
*/
class renderProcessor : public objectProcessor{
private:
	sf::RenderWindow *window;
	viewProcess *view;

	void _Render(vector2 pos, util::Angle &angle , renderData *data, bool centered);
public:

	renderProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager);
	void onObjectAdd(Object *obj);
	void preProcess(){};
	void Process(float dt);
	void onObjectRemove(Object *obj){};
};
