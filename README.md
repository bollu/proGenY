proGenY
=======

A procedurally generated 2d-shooter. 

About
=====

This is a pet project of mine to create a procedurally generated game. It's more of an experimental thing than a build-a-game thing :)

It was started with a vision of 2d borderlands with procedurally generated levels (like a segmented terraria). The bullets / guns are all generated procedurally.

As of now, the plumbing's all there, but I need to write nice code for *content*, which as we all know is the hardest part ;)

This is in *no shape or form* cross-platform (practically). Theoretically, it _should_compile across OS'es 'cause I'm using cross-platform libraries, but Windows has some unique quirks that will surely need some ```#ifdef```s. 


Building / Running
==================

The build system is (slightly) crazy. It uses [Tup](http://gittup.org/tup/), a _blazing fast_ build system. It beats everything I've used (except [Ninja](http://martine.github.io/ninja/), but setting it up is a serious pain). Uses your usual boilerplate Makefiles for building. (Side note - there is no ``` make clean ```, cause Tup is awesome and needs no clean.) 

#To build:#

run ``` make build ```. This builds using Tup and clang. if you do not have either installed, install them and then build. If it spits no errors out, you're done!


#To Run:#

Yes, there's a section on how to run. That's 'cause you can run it in 2 ways - the usual ``` double-click-the-exe ``` or ``` ./a.out ``` method, and the other method, which is to invoke ``` make run ```.

``` make run ``` spawns a new terminal and dumps all the console output in there. I've done it this way 'cause that's how Visual Studio does it _and I like it that way_. It's useful for reading debug logs without making a mess of your main terminal.


Interesting bits of Code:
=========================

The entire game is component-based, so that may be interesting to some people. Look at the folders ```src/core/ObjProcessors/```, ```Object.h/cpp```, ```objectMgr.h/cpp``` for inspiration.

The game uses a very usable implementation of the [Observer pattern](http://en.wikipedia.org/wiki/Observer_pattern) which could be useful to look at.


-----------

The code uses a lot of ideas borrowed from a lot of people. Here's an incomplete list of things that are borrowed / straight out ripped :)

* correct timestep from the [Fix Your Timestep article](http://gafferongames.com/game-physics/fix-your-timestep/)

* the component based system from Evolve your [hierarchy](http://www.gamedev.net/page/resources/_/technical/game-programming/evolve-your-hierarchy-refactoring-game-entities-with-components-r3025)

* the Process system from the [Enginuity Articles](http://www.gamedev.net/page/resources/_/technical/game-programming/enginuity-part-i-r1947) (A fantastic read. some techniques are over-the-top, but others are quite practical. )

* A lot of techniques from the [Procedural Generation Wiki](http://pcg.wikidot.com/) 

* to be implemented [object loader](http://www.gamedev.net/page/resources/_/technical/general-programming/a-simple-c-object-loader-r2698)

* the soon-to-be copied config files from [ppsspp/native](https://github.com/hrydgard/native) 

Thank You's
===========

The project uses [Box2d](http://box2d.org/) for the physics and [SFML](http://www.sfml-dev.org/) for the low-level stuff. A huge thank-you to Erin Catto for the great physics engine, and a lot of love for SFML too <3


[Clang](http://clang.llvm.org/) is used to compile code. It's a brilliant compiler and is *way* ahead of GCC and Visual Studio. Thank you guys for making such an awesome compiler! (I know, I know, this is not equivalent to posting the license.)



