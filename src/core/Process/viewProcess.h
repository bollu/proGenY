#pragma once
#include "Process.h"
#include "processMgr.h"
#include "../include/SFML/Graphics.hpp"
#include "../Messaging/eventMgr.h"
#include "../util/mathUtil.h"
#include "windowProcess.h"

/*!Acts as a wrapper around the projection Matrix.

       Provides conversion functions between Game, Rendering and Screen coordinates

       Game coordinates corresponds to box2d coordinates. all high-level game
       processing is done in game coordinates.
       The origin is bottom left.Positive x axis is rightwards. Positive y axis is upwards

       Rendering coordinates is an intermediate between screen and game coordinates.
       It is a scaled version of the game coordinates. The rendering layers use the rendering
       coordinate system before converting to screen coordinates
       The origin is bottom left. Positive x axis is rightward. Positive y axis is upwards

       The screen coordinates corresponds to actual SDML coordinates. The most low-level layers
       such as the event processing layer and the the final rendering step use screen coordinates.
       All SFML related computation occurs is screen coordinates. anything that has to be used by
       higher level layers *must be converted* to game coordinates.
       The origin is top left. Positive x axis is rightward. Positive y axis is *downwards*.


       Game Coord - box2d coordinates
       View  Coord - box2d coordinates * scaling
       Render Coord - box2d coordinates * scaling + inverted
       Screen Coord - direct 1:1 mapping to screen. (0, 0) to (sceenWidth, screenHeight)



 */
class viewProcess : public Process,
	            public Observer
{
	sf::RenderWindow *window;
	float windowHeight;
	sf::View defaultView;
	float game2RenderScale;
	eventMgr &eventManager;


public:
	viewProcess ( processMgr &processManager, Settings &settings, eventMgr &_eventManager );
	void Update ( float dt );

	/*!converts game coordinates to rendering coordinates

	   @param [in] gameCoord: the coordinates in-game to be converted
	                to the rendering Coordinate system


	 */
	vector2 game2ViewCoord ( vector2 gameCoord );

	/*!converts render coordinates to game coordinates
	   @param [in] gameCoord: the coordinates in the rendering Coordinate system
	         to be converted to the game Coordinate system
	 */
	vector2 view2GameCoord ( vector2 renderCoord );
	/*!converts rendering coordinates to screen coordiantes*/
	vector2 view2RenderCoord ( vector2 renderCoord );
	/*!converts from screen to rendering coordinates*/
	vector2 render2ViewCoord ( vector2 screenCoord );

	/*!converts from screen coordinates to render coordinates
	   \sa viewProcess*/
	vector2 screen2RenderCoord ( vector2 screenCoord );

	/*!converts from rendering coordinates to screen coordinates
	   \sa viewProcess*/
	vector2 render2ScreeenCoord ( vector2 renderCoord );

	/*!moves the viewport by the offset value

	   @param [in] offset: the offset in screen coordinates
	        by which to move the viewport
	 */
	void move ( vector2 offset );

	/*!sets the center of the viewport

	   @param [in] center: the center of the viewport in screen
	                                coordinates
	 */
	void setCenter ( vector2 center );

	/*!sets the angle of the viewport

	   @param [in] angle : sets the angle of the viewport
	 */
	void setRotation ( util::Angle angle );

	/*!returns the center of the viewport

	   \return the center of the viewport in screen coordinates
	 */
	vector2 getCenter ();

	/*!gives the scaling value to convert the game to the rendering
	        coordinate system.

	   \return the amount by which game coordinates are scaled to convert to
	        render coordinates
	 */
	float getGame2RenderScale ();

	/*!gives the scaling value to convert the rendering to the game
	        coordinate system.

	   \return the amount by rendering coordinates are scaled to convert to
	        game coordinates
	 */
	float getRender2GameScale ();
	void  recieveEvent ( const Hash *eventName, baseProperty *eventData );
};