#ifndef _ORDERPARAMETER_H
#define _ORDERPARAMETER_H

#include <Parameters.h>
#include <Grid.h>
#include <Context.h>
#include <OrderParameterComponent.h>

template <typename T>
class OrderParameter : public ContextModule<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
	// Constructors and Destructors
							OrderParameter();
	virtual					~OrderParameter();

							OrderParameter( const OrderParameter<T>& other );
							OrderParameter( Context<T>* ctx );

	OrderParameter<T>&		operator=( OrderParameter<T> right );

	// Functions
	virtual int 			initialize();

	void					add( OrderParameterComponent<T>* delta );

	// Getters
	int							getNumOrderParameterComponents() const;
	OrderParameterComponent<T>*	getOrderParameterComponent( int delta_idx ) const;
	GridGPU*					getOrderParameterComponentField( int delta_idx ) const;

	GridGPU* 				getOrderParameterField();
	GridGPU* 				getOrderParameterFieldRotated( const T angle );									// Perform rotation and return pointer
	GridGPU*				getOrderParameterFieldRotated() const { return _orderParameterCompoundRotated; }		// Return pointer to current rotational frame

	char*					type() const;
	
private:
	template <typename S>
	friend void				swap( OrderParameter<S>& A, OrderParameter<S>& B );

	std::vector<OrderParameterComponent<T>*> _orderParameterComponentList;

	GridGPU*				_orderParameterCompound;
	GridGPU*				_orderParameterCompoundRotated;
};

#endif

