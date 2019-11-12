#include "ParameterIterator.h"

template <typename T>
ParameterIterator<T>::ParameterIterator()
	: _p_min((T)0.0), _p_max((T)0.0), _p_inc((T)1.0)
{
	_sweepDir = 1;
	_p_num = 0;
	begin();
}


template <typename T>
ParameterIterator<T>::ParameterIterator( T p_min, T p_max )
	: _p_min(p_min), _p_max(p_max)
{
	_sweepDir = 1;
	_p_inc = _p_max - _p_min;
	
	if ( p_min != p_max )
		updateNumber();
	else
		setSingleValue();

	begin();
}

template <typename T>
ParameterIterator<T>& ParameterIterator<T>::operator=( const ParameterIterator<T>& right )
{
	if ( &right != this )
	{
		_p_min = right._p_min;
		_p_max = right._p_max;
		_p_inc = right._p_inc;
		_p_num = right._p_num;
	
		_sweepDir = right._sweepDir;
		_count = right._count;
		_isDone = right._isDone;
	}
	
	return *this;
}


template <typename T>
void ParameterIterator<T>::updateIncrement()
{
	if (_p_num == 1)
		_p_inc = _p_max - _p_min;
	else
		_p_inc = (_p_max - _p_min) / ((T)_p_num - 1);
}


template <typename T>
void ParameterIterator<T>::updateNumber()
{
	T num = (_p_max - _p_min) / _p_inc + (T)1.0;
	_p_num = num;
	
	// Avoid potential numerical rounding error
	if ( (T)_p_num - num < -(T)0.999999999 )
		_p_num++;
}

template <>
void ParameterIterator<int>::updateNumber()
{
	int num = (_p_max - _p_min) / _p_inc;

	if ( num * _p_inc < (_p_max - _p_min) )
		num++;
		
	_p_num = num;	
}


template <typename T>
void ParameterIterator<T>::setMin( T p_min )
{
	_p_min = p_min;
	updateNumber();
}


template <typename T>
void ParameterIterator<T>::setMax( T p_max )
{
	_p_max = p_max;
	updateNumber();
}


template <typename T>
void ParameterIterator<T>::setIncrement( T p_inc )
{
	const T range = _p_max - _p_min;
	
	if ( range != (T)0.0 )
	{
		if ( fabs(p_inc) >= fabs(range) )
			_p_inc = range;
		else
			_p_inc = p_inc;
			
		updateNumber();
	}
}


template <typename T>
void ParameterIterator<T>::setNum( int p_num )
{
	_p_num = p_num;
	updateIncrement();
}


template <typename T>
void ParameterIterator<T>::setSweepDirection( int sweepDir )
{
	_sweepDir = sweepDir;
}


template <typename T>
void ParameterIterator<T>::setNull()
{
	_p_num = 0;
	begin();
}


template <typename T>
void ParameterIterator<T>::setSingle()
{
	_p_min = 0;
	_p_max = 1;
	_p_num = 1;
	begin();
}


template <typename T>
void ParameterIterator<T>::setSingleValue()
{
	_p_max = _p_min + _p_inc;
	_p_num = 1;
	begin();
}


template <typename T>
void ParameterIterator<T>::begin( int idx )
{
	_count = idx;
	_isDone = 0;
	
	if ( idx >= _p_num )
	{
		_count = _p_num;
		_isDone = 1;
	}
}


template <typename T>
T ParameterIterator<T>::next( int step )
{
	_count += step;

	if ( _count >= _p_num )
		_isDone = 1;
		
	return getCurrentValue();
}


template <typename T>
T ParameterIterator<T>::getCurrentValue() const
{
	if ( _sweepDir == 1 )
		return _p_min + _p_inc*(T)(_count - _isDone);
	else
	{
		T p_max = _p_min + _p_inc*(T)(_p_num-1);
		
		return p_max - _p_inc*(T)(_count - _isDone);
	}
}


template <typename T>
T ParameterIterator<T>::getNumInterval() const
{
	if ( getCurrentValue() + getIncrement() >= getMax() )
		return getMax() - getCurrentValue();
	else
		return getIncrement();
}



