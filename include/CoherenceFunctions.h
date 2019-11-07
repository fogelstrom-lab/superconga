#ifndef _COHERENCEFUNCTIONS_H
#define _COHERENCEFUNCTIONS_H

#include <Grid.h>
#include <Context.h>
#include <ContextModule.h>

template <typename T>
class CoherenceFunctions : public ContextModule<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
				CoherenceFunctions();
	virtual 	~CoherenceFunctions();

				CoherenceFunctions( Context<T>* ctx, Type::coherence cf_type );

	virtual int	initialize();

	int 			getNumComponents() const {return _Nrow*_Ncol;}
	Type::coherence getCoherenceFunctionSymmetry() const {return _cf_type;}

	GridGPU*	getPositive(int row=0, int col=0) const {return _pos[row*_Ncol+col];}
	GridGPU*	getNegative(int row=0, int col=0) const {return _neg[row*_Ncol+col];}

	GridGPU*	getPositive(int idx) const {return _pos[idx];}
	GridGPU*	getNegative(int idx) const {return _neg[idx];}

private:
	Type::coherence 		_cf_type;

	// 1x1 for singlet (scalar), 2x2 for triplet (2x2 matrix)
	const int 				_Nrow;
	const int 				_Ncol;

	std::vector<GridGPU*>	_pos;		// gamma
	std::vector<GridGPU*>	_neg;		// gammabar
};

#endif
