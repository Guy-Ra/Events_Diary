from events_diary.__main__ import main


def test_main_prints_success_message(capsys) -> None:
    main()

    captured = capsys.readouterr()

    assert captured.out == "Events Diary initialized successfully.\n"