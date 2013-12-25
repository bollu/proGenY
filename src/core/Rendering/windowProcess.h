#pragma once
#include "../controlFlow/Process.h"
#include "../controlFlow/processMgr.h"
#include "../include/SFML/Graphics.hpp"


/*!represents the window. 
Is responsible for double buffering*/  
class windowProcess : public Process{
	sf::RenderWindow *window;
	sf::Color clearColor;
public:
	windowProcess(processMgr &processManager, Settings &settings, EventManager &eventManager);

	/*!clears the window and gets it ready for rendering */
	virtual void preUpdate();
	
	/*!flips the buffer and renders the back buffer */
	void postDraw();

	void Shudown();
	
	/*!returns the instance of the SFML Window*/
	sf::RenderWindow *getWindow();

	/*!sets the background color of the window on refreshing*/
	void setClearColor(sf::Color color);

};
