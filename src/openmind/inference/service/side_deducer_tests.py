from openmind.inference.service.side_deducer import SideDeducer


def moving(piece, color, rows, columns=0):
    return {
        "piece at source": piece,
        "color at source": color,
        "rows from source to target": rows,
        "columns from source to target": columns,
    }


def a_game():
    """Legal actions of two players, where pawns only ever advance and rooks go both ways."""
    return [
        (moving("pawn", "white", 1), "white"),
        (moving("pawn", "white", 1, 1), "white"),
        (moving("rook", "white", 3), "white"),
        (moving("rook", "white", -2), "white"),
        (moving("pawn", "black", -1), "black"),
        (moving("pawn", "black", -1, -1), "black"),
        (moving("rook", "black", -4), "black"),
        (moving("rook", "black", 2), "black"),
    ]


def in_two_positions(examples):
    return examples + examples, [0] * len(examples) + [1] * len(examples)


def test_a_value_only_one_player_ever_moves_belongs_to_that_player():
    examples, positions = in_two_positions(a_game())

    sides = SideDeducer().deduce(examples, positions)

    assert sides.whose("piece", "pawn") is None
    assert sides.whose("color", "white") == "white"
    assert sides.whose("color", "black") == "black"


def test_a_player_faces_the_way_their_one_way_piece_goes():
    examples, positions = in_two_positions(a_game())

    sides = SideDeducer().deduce(examples, positions)

    assert sides.toward("white") == 1
    assert sides.toward("black") == -1


def test_a_game_whose_pieces_all_move_both_ways_has_no_side_to_tell():
    examples = [
        (moving("rook", "white", 3), "white"),
        (moving("rook", "white", -2), "white"),
        (moving("rook", "black", -4), "black"),
        (moving("rook", "black", 2), "black"),
    ]

    sides = SideDeducer().deduce(*in_two_positions(examples))

    assert sides.facing == ()
    assert sides.toward("white") == 0


def test_a_piece_seen_going_one_way_in_a_single_position_says_nothing():
    """A rook that happened to move up twice is as one-way as a pawn, until another position disagrees."""
    examples = [(moving("rook", "white", 3), "white"), (moving("rook", "white", 2), "white")]

    sides = SideDeducer().deduce(examples, [0, 0])

    assert sides.facing == ()


def test_a_value_both_players_move_belongs_to_neither():
    examples = [
        (moving("boulder", "grey", 1), "white"),
        (moving("boulder", "grey", -1), "black"),
    ]

    sides = SideDeducer().deduce(examples, [0, 1])

    assert sides.whose("piece", "boulder") is None
