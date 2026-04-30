// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

namespace search {

// This interface represents a search strategy which assumes:
// - The buffer is sorted
// - The length of the buffer is greater than 1
// - The searched element is greater than the first element
// - The searched element is smaller than the last element
//
// It returns the index of the first element greater than or equal
// to the searched element, i.e. implements std::lower_bound().
interface ISearchStrategy
{
    associatedtype Value : IComparable;
    associatedtype Index : IInteger;

    static Index search(Value value, ImmutablePtr<Value> data, Index length);
};

struct LinearSearch<_Value : IComparable, _Index : IInteger> : ISearchStrategy
{
    typedef _Value Value;
    typedef _Index Index;

    static Index search(Value value, ImmutablePtr<Value> data, Index length)
    {
        for (Index i = Index(0); i < length; i = i + Index(1)) {
            if (!(data[i.toUInt()] < value)) {
                return i;
            }
        }
        return length;
    }
};

struct BinarySearch<_Value : IComparable, _Index : IInteger> : ISearchStrategy
{
    typedef _Value Value;
    typedef _Index Index;

    static Index search(Value value, ImmutablePtr<Value> data, Index length)
    {
        Index count = length;
        Index first = Index(0);
        while (count > Index(0)) {
            const Index step = count / Index(2);
            const Index it = first + step;
            if (data[it.toUInt()] < value) {
                first = it + Index(1);
                count = count - (step + Index(1));
            }
            else {
                count = step;
            }
        }
        return first;
   }
};


// This function searches in a sorted list for the pair of values surrounding a searched element,
// along whether the value is included in the list range or not.
// If the length is 0, (-1,-1, false) is returned.
// If the searched element is smaller than the first value, (0,0,false) is returned.
// If the searched element is greater than the last value (length-1,length-1,false) is returned.
// If the searched element is equal to the value at index i, (i,i,true) is returned.
// Otherwise, (i,i+1,true) is returned where i is the index of the value just below the
// searched element and i+1 is the value just above.
__generic<SearchStrategy : ISearchStrategy>
Tuple<SearchStrategy.Index, SearchStrategy.Index, bool> find_surrounding_indices(
    SearchStrategy.Value value, ImmutablePtr<SearchStrategy.Value> data, SearchStrategy.Index length
) {
    typedef SearchStrategy.Index Index;
    typedef SearchStrategy.Value Value;

    static_assert(__isSignedInt<Index>(), "Index type must be signed");

    if (length < Index(1)) {
        return makeTuple(Index(-1), Index(-1), false);
    }

    const var first_value = data[0];
    if (value <= first_value) {
        return makeTuple(Index(0), Index(0), value == first_value);
    }

    // We know that length >= 1 at this point, so ok to go unsigned.
    const Index last_index = length - Index(1);
    const var last_value = data[last_index.toUInt()];
    if (value >= last_value) {
        return makeTuple(last_index, last_index, value == last_value);
    }

    // At this point, we know that we have more than one element, otherwise the
    // value would have been either lesser, equal or greater than the single
    // value and we would have returned already.
    // So it's safe to call the search strategy.
    const Index index_after = SearchStrategy.search(value, data, length);

    // Check for equality.
    if (data[index_after.toUInt()] == value) {
        return makeTuple(index_after, index_after, true);
    }

    return makeTuple(index_after - Index(1), index_after, true);
}

} // namespace search
