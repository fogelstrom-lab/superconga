#ifndef _PLOTREGISTRATION_H
#define _PLOTREGISTRATION_H

#include <Grid.h>

template <typename DEV>
class PlotRegistration
{
	PlotRegistration();

public:
	PlotRegistration( int N ) : gridList(new DEV*[N]), typeList(new Type::type[N]), fieldList(new int[N]) {}
	~PlotRegistration() { delete[] gridList; delete[] typeList, delete[] fieldList; }

	DEV*		getGrid( int idx ) { return gridList[idx]; }
	Type::type	getType( int idx ) { return typeList[idx]; }
	int			getField( int idx ) { return fieldList[idx]; }

	void registerPlot( DEV* grid, Type::type representation, int field, int idx ) {
		gridList[idx] = grid;
		typeList[idx] = representation;
		fieldList[idx] = field;
	}

	void getPlotInfo( DEV*& grid, Type::type& representation, int& field, int idx ) {
		grid = gridList[idx];
		representation = typeList[idx];
		field = fieldList[idx];
	}

	DEV**		gridList;
	Type::type*	typeList;
	int*		fieldList;
};

#endif
