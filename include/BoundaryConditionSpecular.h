#ifndef _BOUNDARYCONDITIONSPECULAR_H
#define _BOUNDARYCONDITIONSPECULAR_H

#include <BoundaryCondition.h>

template <typename T>
class BoundaryConditionSpecular : public BoundaryCondition<T>
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

public:
					BoundaryConditionSpecular() : BoundaryCondition<T>() {}
	virtual			~BoundaryConditionSpecular() {}

					//BoundaryConditionSpecular( Context<T>* ctx ) : BoundaryCondition<T>(ctx) {}
					
	virtual void	computeBoundaryCondition( Context<T>* ctx );

private:
	
};

#endif
