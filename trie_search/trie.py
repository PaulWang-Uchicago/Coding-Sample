from typing import Any, Iterable
from collections.abc import MutableMapping


def character_to_key(char: str) -> int:
    """
    Given a character return a number between [0, 26] inclusive.

    Letters a-z should be given their position in the alphabet 0-25, regardless of case:
        a/A -> 0
        z/Z -> 25

    Any other character should return 26.
    """

    # IMPLEMENTATION HERE
    if "a" <= char <= "z":
        return ord(char) - ord("a")
    elif "A" <= char <= "Z":
        return ord(char) - ord("A")
    else:
        return 26


class Trie(MutableMapping):
    """
    Implementation of a trie class where each node in the tree can
    have up to 27 children based on next letter of key.
    (Using rules described in character_to_key.)

    Must implement all required MutableMapping methods,
    as well as wildcard_search.
    """

    class Node:
        """
        Class Node contains children, value, and is_end attributes.
        Children is a placeholder for the next letter in the key.
        """

        def __init__(self):
            self.children = [None] * 27
            self.value = None
            self.is_end = False

    def __init__(self):
        self.root = self.Node()
        self.size = 0

    def __getitem__(self, key: str) -> Any:
        """
        Given a key, return the value associated with it in the trie.

        If the key has not been added to this trie, raise `KeyError(key)`.
        If the key is not a string, raise `ValueError(key)`
        """

        # First check if key is a string, raise KeyError if not
        if not isinstance(key, str):
            raise KeyError(key)

        # Traverse the trie to find the value associated with the key
        node = self.root
        for char in key:
            key = character_to_key(char)
            if node.children[key] is None:
                raise KeyError(key)

            node = node.children[key]

        # If the node is None or not the end of the key, raise KeyError
        if node is None or not node.is_end:
            raise KeyError(key)

        return node.value

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Given a key and value, store the value associated with key.

        Like a dictionary, will overwrite existing data if key already exists.

        If the key is not a string, raise `ValueError(key)`
        """

        # First check if key is a string, raise KeyError if not
        if not isinstance(key, str):
            raise KeyError(key)

        # Traverse the trie to find the value associated with the key
        node = self.root
        for char in key:
            key = character_to_key(char)
            if node.children[key] is None:
                node.children[key] = Trie.Node()
            node = node.children[key]

        # If the node is not the end of the key, increment size
        if not node.is_end:
            self.size += 1

        # Set the value and mark the node as the end of the key
        node.is_end = True
        node.value = value

    def __delitem__(self, key: str) -> None:
        """
        Remove data associated with `key` from the trie.

        If the key is not a string, raise `ValueError(key)`
        """

        # First check if key is a string, raise KeyError if not
        if not isinstance(key, str):
            raise ValueError(key)

        # Traverse the trie to find the value associated with the key
        node = self.root
        for char in key:
            key = character_to_key(char)
            if node.children[key] is None:
                raise KeyError(key)
            node = node.children[key]

        # If the node is not the end of the key, raise KeyError
        if not node.is_end:
            raise KeyError(key)

        # Mark the node as not the end of the key and decrement size
        node.is_end = False
        self.size -= 1

    def __len__(self) -> int:
        """
        Return the total number of entries currently in the trie.
        """
        return self.size

    def __iter__(self) -> Iterable[tuple[str, Any]]:
        """
        Return an iterable of (key, value) pairs for every entry in the trie in alphabetical order.
        """

        def _iter(node, prefix: str):
            # Yield the key and value of the node if it is not None
            if node.value is not None:
                yield (prefix, node.value)
            # Iterate through the children of the node
            for i, child in enumerate(node.children):
                if child is not None:
                    # Get the character associated with the child
                    char = chr(i + ord("a")) if i < 26 else "#"
                    # Yield the key and value of the child
                    yield from _iter(child, prefix + char)

        return _iter(self.root, "")

    def wildcard_search(self, key: str) -> Iterable[tuple[str, Any]]:
        """
        Search for keys that match a wildcard pattern where a '*' can represent any single character.

        For example:
            - c*t would match 'cat', 'cut', 'cot', etc.
            - ** would match any two-letter string.

        Returns: Iterable of (key, value) pairs meeting the given condition.
        """

        def _wildcard_search(node, key, index, prefix):
            # If the index is equal to the length of the key, yield the prefix and value of the node
            if index == len(key):
                if node.is_end:
                    yield (prefix, node.value)
                return

            char = key[index]
            # If the character is a wildcard, iterate through the children of the node
            if char == "*":
                for i, child in enumerate(node.children):
                    if child is not None:
                        # Get the next character and yield the key and value of the child
                        next_char = chr(i + ord("a")) if i < 26 else "_"
                        yield from _wildcard_search(
                            child, key, index + 1, prefix + next_char
                        )
            # If the character is not a wildcard, get the key index and continue searching
            else:
                key_index = character_to_key(char)
                if node.children[key_index] is not None:
                    yield from _wildcard_search(
                        node.children[key_index], key, index + 1, prefix + char
                    )

        return _wildcard_search(self.root, key, 0, "")
