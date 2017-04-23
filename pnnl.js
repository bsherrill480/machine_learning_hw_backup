// Non JS Changes:
//   CSS change:
//     added class .header-floater to header, i.e.
//     header, .header-floater {
//         font-size: larger;
//         font-weight: bold;
//         float: left;
//     }

var solution = [81, 50, 57, 117, 90, 51, 74, 104, 100, 72, 86, 115, 89, 88, 82, 112, 98, 50, 53, 122, 73, 81, 61, 61],
    c = function(input) {
        return atob(input.reduce(
            function(acc, val) {
                return acc + String.fromCharCode(val);
            },
            ""
        ));
    };

function makeHeaderCharacterSpan(content, characterIndex) {
    return '<span class="header-character" id="header-character-' + characterIndex + '">' + content + "</span>";
}

// Add a new character span to the header with an empty space
function growHeaderCharacters($header) {
    var numCharacterHeaders = $header.find('.header-character').length;
    $header.append(makeHeaderCharacterSpan('&nbsp', numCharacterHeaders));
}

$(function() {
    var $section = $('section'),
        $header = $('header'),
        sectionIndex = 0,
        solutionIndex = 0,
        charCounts = {},
        started = 0,
        ended = 0;

    // compute correct counts (ignoring solution)
    $section.text().split('').forEach(function(character) {
        character = character.toLowerCase();
        // note: (undefined + 1) || 1 => 1
        charCounts[character] = (charCounts[character] + 1) || 1;
    });

    // decode solution
    solution = c(solution);

    // correct counts with solution
    solution.split('').forEach(function(c) {
        c = c.toLowerCase();
        // don't decrement if character doesn't exist;
        if(typeof charCounts[c] !== 'undefined') {
            charCounts[c]--;
        }
    });

    // depends on and changes the following variables:
    //     sectionIndex,
    //     solutionIndex,
    //     charCounts,
    //     started,
    //     ended
    // Iterates through section and counts the letters that are not part of the solution.
    // The UI floats the letters to their appropriate location.
    var runCount = function() {
        var currentCharacter, $countBox;
        // test if we're finished
        if(sectionIndex < $section.html().length)
        {
            // what is the next character in the input?
            currentCharacter = $section.html()[sectionIndex];
            $countBox = null; // box to count characters that are not part of solution

            var $floater = $("<span>" + currentCharacter + "</span>"),
                isTop = currentCharacter === solution[solutionIndex];

            // move the character to the top or the bottom
            if(isTop)
            {
                if(!solutionIndex) $header.html('');

                // make room for the new character
                growHeaderCharacters($header);
                solutionIndex++; // increment the solution counter
            }
            else if(currentCharacter.match(/[A-Za-z]/)) // we're going to count all the alpha characters
            {
                currentCharacter = currentCharacter.toLowerCase(); // ignore the case when counting
                $countBox = $("#" + currentCharacter + " .count"); // get the element that contains the count

                if(!$countBox.length) // create it if it doesnt exist
                {
                    $("footer").append(
                        "<div id='" + currentCharacter + "'>" +
                        "   <div>" + currentCharacter + "</div>" +
                        "   <div class='count'>0</div>" +
                        "</div>"
                    );
                    // get newly created count box
                    $countBox = $("#" + currentCharacter + " .count");
                }
            }

            // if we have an animating character, do the animation
            if(isTop || $countBox)
            {
                // add a marking element for the start position of the animation
                $section.html(
                    $section.html().substr(0,sectionIndex) +
                    "<span id='start" + sectionIndex + "'>" +
                    (currentCharacter.match(/([^A-Za-z])|./)[1] || "_") +
                    "</span>" + $section.html().substr(sectionIndex+1)
                );

                var $start = $("#start" + sectionIndex);
                var rect = $start[0].getBoundingClientRect(); // get the px position of the start element
                $floater
                    .css("top", rect.top + "px")
                    .css("left", rect.left + "px");
                $(".floaters").append($floater); // add an element to be the moving character
                $start.prop("outerHTML", $start.text()); // remove the start element now that we have the position

                // started another animation
                started++;

                // The following if and else do:
                // 1) find the end position for the animation
                // 2) set the new position and let css transitions do the rest
                // 3) adjust the styles to match the end result
                var floaterWidth = $floater.width();
                if(isTop) {
                    rect = $header[0].getBoundingClientRect();
                    $floater
                        .addClass('header-floater')
                        .css("top", rect.top + "px")
                        .css("left", rect.right - floaterWidth + "px");
                }
                else {
                    rect = $("#" + currentCharacter + " div:first-child")[0].getBoundingClientRect();
                    var rectWidth = rect.right - rect.left;
                    var floaterFinalDestination = rect.left + (rectWidth / 2) - (floaterWidth / 2);
                    $floater
                        .css("top", rect.top + "px")
                        .css("left", floaterFinalDestination + "px");
                }

                // capture current state of variables, since variables like solutionIndex
                // continue to be modified while waiting for setTimeout to run it's function
                // solutionIndex for the current character is 1 less,
                // because we incremented earlier
                var localSolutionIndexForCurrentChar = solutionIndex - 1;

                // remove the floating character when the transition ends
                setTimeout(function() {
                    $floater.remove();

                    if(isTop)
                    {
                        $header.find('#header-character-' + localSolutionIndexForCurrentChar).text(currentCharacter);
                    }
                    else if($countBox)
                    {
                        // increment the count and redraw the bar with a new height
                        $countBox
                            .html(Number($countBox.html()) + 1)
                            .css("height", $countBox.html() + "px");
                    }

                    // finished another animation
                    ended++;

                    if(started === ended)
                    {
                        finalChecks();
                    }
                }, 1000);
            }

            sectionIndex++;

            setTimeout(runCount, 10);
        }
    };

    var finalChecks = function() {

        console.log("<header> text === solution", $header.text(), solution, $header.text() === solution);
        console.log("No extra elements?", !$(".floaters").children().length && !$section.children().length);
        console.group("Correct character counts?");
        $('footer .count').each(function(i, e) {
            e = $(e);
            console.log(e.parent().attr('id'), +e.text(), charCounts[e.parent().attr('id')], +e.text() === charCounts[e.parent().attr('id')]);
        });
        console.groupEnd();
    };

    runCount();
});
