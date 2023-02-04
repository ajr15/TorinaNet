from typing import Tuple, Any, Iterator
from abc import ABC, abstractclassmethod

class HashedCollection:

    @abstractclassmethod
    def add(self, obj):
        """Method to add new object to the collection"""
        pass

    @abstractclassmethod
    def delete(self, obj):
        """Method to delete an object in the collection. 
        RETURNS: if object is in collection, deletes it else, raises KeyError"""
        pass

    @abstractclassmethod
    def get(self, obj):
        """Method to get an object from the collection. 
        RETURNS: if object is in collection, returns it (from the collection) else, raises KeyError"""
        pass

    @abstractclassmethod
    def get_key(self, obj) -> int:
        """Method to get the hash-key of an object"""
        pass

    @abstractclassmethod
    def has(self, obj) -> bool:
        """Method to check if a collection has an object"""
        pass

    @abstractclassmethod
    def objects(self) -> Iterator[Any]:
        """Iterate over all objects in collection"""
        pass

    @abstractclassmethod
    def keys(self) -> Iterator[int]:
        """Iterate over all hash-keys in a collection"""
        pass
    
    @abstractclassmethod
    def to_dict(self) -> dict:
        """Export the arguments of the collection to a dictionary"""
        pass

    def items(self) -> Iterator[Tuple[int, Any]]:
        """Iterator over all items in the collection"""
        return zip(self.keys(), self.objects())

    def union(self, collection):
        """Unite two hashed-collections. adding new collection's objects to the current collection."""
        if not isinstance(collection, HashedCollection):
            raise ValueError("Can unite only two HashedCollection instances")
        for obj in collection.objects():
            self.add(obj)

    def substract(self, collection) -> Iterator[Any]:
        """Substract two hashed-collections to find different objects between them. returning an iterator over the different objects between current collection and provided collection (objects in current and not in provided)"""
        if not isinstance(collection, HashedCollection):
            raise ValueError("Can substract only two HashedCollection instances")
        for obj in self.objects():
            if not collection.has(obj):
                yield obj

    def intersect(self, collection) -> Iterator[Any]:
        """find shared objects between two hashed-collections. returning an iterator over the shared objects between current collection and provided collection"""
        if not isinstance(collection, HashedCollection):
            raise ValueError("Can substract only two HashedCollection instances")
        for obj in self.objects():
            if collection.has(obj):
                yield obj

    def issubset(self, collection) -> bool:
        """find if current collection is a subset of the provided collection"""
        if not isinstance(collection, HashedCollection):
            raise ValueError("Can substract only two HashedCollection instances")
        return all([collection.has(obj) for obj in self.objects()])

