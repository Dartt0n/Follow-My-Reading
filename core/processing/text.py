
def match(first_text: str, second_text: str):

    # Returns a list of changes that need to be made to the first text to get the second one
    # The output format is:
    # List[Tuple(Index in the first text where the difference was found,
    #            The segment of the first text which is to be removed,
    #            The segment of the second text which is to be substituted in)]

    # Collections of all similar symbols for the purposes of loosening the strictness
    similar = ["“\"'’”‘", " \n"]

    for i in similar:
        for j in i[1:]:
            first_text = first_text.replace(j, i[0])
            second_text = second_text.replace(j, i[0])

    lev_dp = [[461782368126487236] * (len(second_text) + 1) for i in range(len(first_text) + 1)]
    lev_dp[0][0] = 0
    for i in range(1, len(first_text) + 1):
        lev_dp[i][0] = i
    for i in range(1, len(second_text) + 1):
        lev_dp[0][i] = i
    for i in range(1, len(first_text) + 1):
        for j in range(1, len(second_text) + 1):
            lev_dp[i][j] = min(lev_dp[i-1][j-1] + (first_text[i-1] != second_text[j-1]), lev_dp[i-1][j] + 1, lev_dp[i][j-1] + 1)

    result = []
    curx = len(first_text)
    cury = len(second_text)
    while curx * cury != 0:
        if first_text[curx-1] == second_text[cury-1]:
            result.append(first_text[curx-1])
            curx -= 1
            cury -= 1
            continue
        optimal = min(lev_dp[curx - 1][cury - 1], lev_dp[curx - 1][cury], lev_dp[curx][cury - 1])
        if optimal == lev_dp[curx - 1][cury - 1]:
            result.append(first_text[curx - 1] + '-' + second_text[cury - 1])
            curx -= 1
            cury -= 1
        elif optimal == lev_dp[curx - 1][cury]:
            result.append(first_text[curx - 1] + '-_')
            curx -= 1
        elif optimal == lev_dp[curx][cury - 1]:
            result.append("_-" + second_text[cury - 1])
            cury -= 1

    if curx != 0:
        result.append(first_text[:curx] + "-_")
    elif cury != 0:
        result.append("_-" + second_text[:cury])

    joined_result = []
    for i in result[::-1]:
        if not joined_result:
            if "-" in i:
                joined_result.append(i.split("-"))
            else:
                joined_result.append(i)
        elif type(joined_result[-1]) == list:
            if "-" in i:
                joined_result[-1][0] += i[0]
                joined_result[-1][1] += i[2]
            else:
                joined_result.append(i)
        else:
            if "-" in i:
                joined_result.append(i.split("-"))
            else:
                joined_result[-1] += i
    for i in joined_result:
        if type(i) == list:
            i[0] = i[0].replace("_", "")
            i[1] = i[1].replace("_", "")

    answer = []
    first_index = 0
    for i in joined_result:
        if type(i) == list:
            answer.append((first_index, i[0], i[1]))
            first_index += len(i[0])
        else:
            first_index += len(i)
    return answer


print(match(
        """
        On seeing the Russian general he threw back his head, with its long hair curling to his shoulders, in a majestically royal manner, and looked inquiringly at the French colonel. The colonel respectfully informed His Majesty of Balashev’s mission, whose name he could not pronounce.
        """,
        """
        On seeing the Russian general he threw back his head, with its long hair curling to his shoulders, in a
majestically royal manner, and looked inquiringly at the French colonel. The colonel respectfully
informed His Majesty of Balashev’s mission, whose name he could not pronounce.
        """
    )
)
