#pragma once


class State;

/*!used to Save and Load State objects

   This class helps save and load a particular State. each State should have a
   corresponding stateSaveLoader that knowns how to save and load the State.
   It ensures a separation of responsibilities
 */
class stateSaveLoader
{
public:
	/*!returns whether the stateSaveLoader done saving
	 */
	bool isDoneSaving (){
		return (this->doneSaving);
	}


	/*!returns whether the stateSaveLoader done loading
	 */
	bool isDoneLoading (){
		return (this->doneLoading);
	}


	virtual void Save () = 0;
	virtual void Load () = 0;

	virtual ~stateSaveLoader (){}


protected:
	stateSaveLoader ( State *state ){
		this->doneSaving = this->doneLoading = false;
	}


	bool doneSaving;
	bool doneLoading;
};