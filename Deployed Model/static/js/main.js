$(document).ready(function () {

    $("#imageUpload").change(function () {
        $(".image-section").fadeIn();
        $("#result-card").hide();
        readURL(this);
    });

    $("#btn-predict").click(function () {

        var form_data = new FormData($("#upload-file")[0]);

        $("#btn-predict").hide();
        $(".loader").show();

        $.ajax({

            type: "POST",

            url: "/predict",

            data: form_data,

            contentType: false,

            cache: false,

            processData: false,

            success: function (data) {

                $(".loader").hide();

                $("#btn-predict").show();

                $("#result-card").fadeIn();

                //--------------------------------
                // Main Result
                //--------------------------------
                let badgeColor = "success";

                if (data.confidence < 40)
                    badgeColor = "danger";
                else if (data.confidence < 70)
                    badgeColor = "warning";
                else
                    badgeColor = "success";

                $("#result").html(
                    "<h3>" + data.stage + "</h3>" +
                    "<h5>Confidence : <span class='badge badge-" +
                    badgeColor +
                    "'>" +
                    data.confidence +
                    "%</span></h5>"
                );


                //--------------------------------
                // Diagnosis Summary
                //--------------------------------

                $("#summary-card").fadeIn();

                $("#summary-stage").text(data.stage);

                $("#summary-confidence").text(data.confidence + "%");

                let risk = "";

                if (data.stage.includes("No Diabetic")) {

                    risk = "🟢 Low";

                }

                else if (data.stage.includes("Mild")) {

                    risk = "🟡 Mild";

                }

                else if (data.stage.includes("Moderate")) {

                    risk = "🟠 Moderate";

                }

                else if (data.stage.includes("Severe")) {

                    risk = "🔴 High";

                }

                else {

                    risk = "🚨 Critical";

                }

                $("#summary-risk").text(risk);

                //--------------------------------
                // Prediction Reliability
                //--------------------------------

                let reliability = "";

                $("#low-confidence-warning").hide();

                if (data.confidence >= 80) {

                    reliability = "🟢 High";

                }

                else if (data.confidence >= 60) {

                    reliability = "🟡 Moderate";

                }

                else {

                    reliability = "🔴 Low";

                    $("#low-confidence-warning").fadeIn();

                }

                $("#summary-reliability").text(reliability);

                //--------------------------------
                // Result Card Color
                //--------------------------------

                if (data.stage.includes("No Diabetic")) {

                    $(".result-card").css("border-left", "6px solid #2ecc71");

                }

                else if (data.stage.includes("Mild")) {

                    $(".result-card").css("border-left", "6px solid #f1c40f");

                }

                else if (data.stage.includes("Moderate")) {

                    $(".result-card").css("border-left", "6px solid #e67e22");

                }

                else if (data.stage.includes("Severe")) {

                    $(".result-card").css("border-left", "6px solid #e74c3c");

                }

                else {

                    $(".result-card").css("border-left", "6px solid #8e44ad");

                }

                //--------------------------------
                // Prediction Probabilities
                //--------------------------------

                let html = "<h5 class='mt-4'>Prediction Probabilities</h5>";

                const sortedProbabilities = Object.entries(data.probabilities)
                    .sort((a, b) => b[1] - a[1]);

                for (const [label, value] of sortedProbabilities) {

                    let color = "#95a5a6";

                    if (label.includes("No Diabetic"))
                        color = "#2ecc71";
                    else if (label.includes("Mild"))
                        color = "#f1c40f";
                    else if (label.includes("Moderate"))
                        color = "#e67e22";
                    else if (label.includes("Severe"))
                        color = "#e74c3c";
                    else if (label.includes("Proliferative"))
                        color = "#8e44ad";

                    html += `

                    <div style="margin-bottom:15px;">

                        <div style="display:flex;justify-content:space-between;">

                            <span>${label}</span>

                            <b>${value}%</b>

                        </div>

                        <div style="background:#ddd;border-radius:20px;height:14px;overflow:hidden;">

                            <div style="width:${value}%;background:${color};height:100%;transition:1s;"></div>

                        </div>

                    </div>

                    `;

                }

                $("#probability-section").html(html);

                //--------------------------------
                // AI Assistant
                //--------------------------------

                $("#ai-info").text(data.info);

                $("#ai-advice").text(data.advice);

                $("#ai-precautions").empty();

                data.precautions.forEach(function (item) {

                    $("#ai-precautions").append("<li>" + item + "</li>");

                });

                $("#ai-diet").empty();

                data.diet.forEach(function (item) {

                    $("#ai-diet").append("<li>" + item + "</li>");

                });

            },

            error: function () {

                $(".loader").hide();

                $("#btn-predict").show();

                alert("Prediction Failed.");

            }

        });

    });

});

function readURL(input) {

    if (input.files && input.files[0]) {

        var reader = new FileReader();

        reader.onload = function (e) {

            $("#imagePreview").css(

                "background-image",

                "url(" + e.target.result + ")"

            );

        };

        reader.readAsDataURL(input.files[0]);

    }

}