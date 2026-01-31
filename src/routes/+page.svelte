<script lang="ts">
	import { gradClassActivationMap } from './cam';
	import * as tf from '@tensorflow/tfjs';
	import { onMount } from 'svelte';
	import { fade } from 'svelte/transition';

	let inputText: HTMLDivElement;
	let predictionsResults: Float32Array[] = [];
	let imgDisplays: string[];
	let files: FileList;
	let uploadedFiles: FileList;
	let predictionsLoading: Boolean[] = [];
	let model: tf.LayersModel;
	let highests: number[] = [];
	let imageElements: HTMLImageElement[] = [];

	const isHighest = (i: number, j: number, a: number, b: number) => {
		return (
			predictionsResults[i][j] > predictionsResults[i][a] &&
			predictionsResults[i][j] > predictionsResults[i][b]
		);
	};

	$: if (uploadedFiles && inputText) {
		if (uploadedFiles.length == 0) inputText.textContent = 'Pilih..';
		else inputText.textContent = uploadedFiles.length + ' file terpilih.';
	}

	function blobToImage(url: string) {
		return new Promise((resolve, reject) => {
			const img = new Image();
			img.onload = () => {
				URL.revokeObjectURL(url);
				resolve(img);
			};
			img.onerror = (error) => {
				URL.revokeObjectURL(url);
				reject(error);
			};
			img.src = url;
		});
	}

	onMount(async () => {
		model = await loadModel();
	});

	const generateGradCAM = async (
		model: tf.LayersModel,
		imageElement: HTMLImageElement,
		classIdx: number
	) => {
		const imgTensor = preprocessImage(imageElement);
		const colored = gradClassActivationMap(model, classIdx, imgTensor);

		const canvas = document.createElement('canvas');
		canvas.width = imageElement.width;
		canvas.height = imageElement.height;

		await tf.browser.toPixels(
			tf.image.resizeBilinear((colored as any).squeeze().toFloat().div(tf.scalar(255)), [
				imageElement.height,
				imageElement.width
			]),
			canvas
		);
		const ctx = canvas.getContext('2d')!;

		ctx.globalAlpha = 0.2;
		ctx.drawImage(imageElement, 0, 0, imageElement.width, imageElement.height);

		canvas.toBlob((blob) => {
			if (!blob) return;
			const url = URL.createObjectURL(blob);
			const link = document.createElement('a');
			link.href = url;
			link.download = `gradcam_${Date.now()}.png`;
			link.click();
			URL.revokeObjectURL(url);
		}, 'image/png');

		tf.dispose([imgTensor, colored]);
	};

	const loadModel = async () => {
		const model = await tf.loadLayersModel('/cancer_detect/model.json');
		console.log('Loaded model successfully!');
		return model;
	};

	const preprocessImage = (imageElement: HTMLImageElement) => {
		const imageTensor = tf.browser.fromPixels(imageElement);
		const resizedImageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);
		const batchedImageTensor = tf.expandDims(resizedImageTensor, 0);
		return batchedImageTensor;
	};

	const classifyImage = async (model: tf.LayersModel, preprocessedImage: tf.Tensor<tf.Rank>) => {
		const predictions = model.predict(preprocessedImage);
		return predictions;
	};

	const predictionImage = async () => {
		for (let i = 0; i < imgDisplays.length; ++i) {
			predictionsLoading[i] = true;
			const imageElement = (await blobToImage(imgDisplays[i])) as HTMLImageElement;
			const preprocessedImage = preprocessImage(imageElement);
			predictionsResults[i] = (await (
				(await classifyImage(model, preprocessedImage)) as tf.Tensor<tf.Rank>
			).data()) as Float32Array;
			const { value: _, index: topClassIdx } = predictionsResults[i].reduce(
				(previousValue, currentValue, currentIndex) => {
					if (currentValue > previousValue.value) {
						return { value: currentValue, index: currentIndex };
					} else {
						return previousValue;
					}
				},
				{ value: -Infinity, index: -1 }
			);
			highests[i] = topClassIdx;
			imageElements[i] = imageElement;
			tf.dispose([preprocessedImage]);
			predictionsLoading[i] = false;
		}
	};

	const predict = async () => {
		console.log('Predicting..');
		await predictionImage();
		console.log('Done!');
	};

	const classifyClicked = async () => {
		files = uploadedFiles;
		uploadedFiles = new DataTransfer().files;

		imgDisplays = [];
		predictionsLoading = [];
		for (const file of files) {
			imgDisplays.push(URL.createObjectURL(file));
		}

		await predict();
	};
</script>

<main>
	<header>
		<h1 class="title">CancerDetect</h1>
		<p class="subtext">Klasifikasi kanker pada paru-paru berbasis Deep Learning.</p>
	</header>

	<div class="content">
		{#each imgDisplays as url, i}
			<div class="card" in:fade={{ duration: 600 }} out:fade={{ duration: 200 }}>
				<p>{files[i].name}</p>
				<div class="classification">
					<img class="ct-scan" in:fade={{ duration: 300 }} src={url} alt={files[i].name} />
					<div class="prediction">
						{#if predictionsLoading[i] == false}
							<h2 in:fade={{ duration: 300 }}>Hasil klasifikasi oleh model AI:</h2>
							<div class="result" in:fade={{ duration: 300 }}>
								<h3 class={isHighest(i, 2, 0, 1) ? 'selected' : ''}>
									Normal: <p class={isHighest(i, 2, 0, 1) ? 'selected' : ''}>
										{(predictionsResults[i][2] * 100.0).toFixed(1)}%
									</p>
								</h3>
								<h3 class={isHighest(i, 0, 1, 2) ? 'selected' : ''}>
									Benign/Jinak: <p class={isHighest(i, 0, 1, 2) ? 'selected' : ''}>
										{(predictionsResults[i][0] * 100.0).toFixed(1)}%
									</p>
								</h3>
								<h3 class={isHighest(i, 1, 0, 2) ? 'selected' : ''}>
									Malignant/Ganas: <p class={isHighest(i, 1, 0, 2) ? 'selected' : ''}>
										{(predictionsResults[i][1] * 100.0).toFixed(1)}%
									</p>
								</h3>
							</div>
							<button
								class="gradcam"
								in:fade={{ duration: 300 }}
								on:click={async () => await generateGradCAM(model, imageElements[i], highests[i])}
								><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
									><path
										fill="currentColor"
										d="m12 16l-5-5l1.4-1.45l2.6 2.6V4h2v8.15l2.6-2.6L17 11zm-6 4q-.825 0-1.412-.587T4 18v-3h2v3h12v-3h2v3q0 .825-.587 1.413T18 20z"
									/></svg
								>&nbsp;Download GradCAM
							</button>
						{:else}
							<h2>Loading..</h2>
						{/if}
					</div>
				</div>
			</div>
		{/each}
	</div>

	<div class="navbar">
		<div class="file-select">
			<p class="file-label">CT Scan Paru-Paru</p>
			<input id="input" type="file" accept="image/*" multiple bind:files={uploadedFiles} />
			<label class="file-box" for="input">
				<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
					<path
						fill="currentColor"
						d="M11 16V7.85l-2.6 2.6L7 9l5-5l5 5l-1.4 1.45l-2.6-2.6V16zm-5 4q-.825 0-1.412-.587T4 18v-3h2v3h12v-3h2v3q0 .825-.587 1.413T18 20z"
					/>
				</svg>
				<div bind:this={inputText}>Pilih..</div>
			</label>
		</div>
		<div class="classify">
			<button on:click={classifyClicked}>
				<svg
					class="left-icon"
					xmlns="http://www.w3.org/2000/svg"
					width="24"
					height="24"
					viewBox="0 0 24 24"
					><path
						fill="currentColor"
						d="m19 1l-1.26 2.75L15 5l2.74 1.26L19 9l1.25-2.74L23 5l-2.75-1.25M9 4L6.5 9.5L1 12l5.5 2.5L9 20l2.5-5.5L17 12l-5.5-2.5M19 15l-1.26 2.74L15 19l2.74 1.25L19 23l1.25-2.75L23 19l-2.75-1.26"
					/>
				</svg>
				<div>&nbsp;Klasifikasi&nbsp;</div>
				<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
					><path
						fill="currentColor"
						d="m19 1l-1.26 2.75L15 5l2.74 1.26L19 9l1.25-2.74L23 5l-2.75-1.25M9 4L6.5 9.5L1 12l5.5 2.5L9 20l2.5-5.5L17 12l-5.5-2.5M19 15l-1.26 2.74L15 19l2.74 1.25L19 23l1.25-2.75L23 19l-2.75-1.26"
					/>
				</svg>
			</button>
		</div>
	</div>
</main>

<svelte:head>
	<title>CancerDetect</title>
	<meta name="description" content="Klasifikasi Kanker pada Paru-Paru berbasis AI" />
</svelte:head>

<style lang="scss">
	$primary-color: #73d800;
	$secondary-color: #2b2b2b;
	$tertiary-color: #f4f4f4;
	$quaternary-color: #c4c4c4;

	$mobile: 560px;
	$tablet: 768px;
	$laptop: 1024px;

	main {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
	}

	header {
		padding-top: 4.8rem;

		.title {
			text-align: center;
			font-size: 4.8rem;
			color: $secondary-color;
		}

		.subtext {
			text-align: center;
			font-size: 2rem;
			color: rgba($secondary-color, 0.7);
		}
	}

	.content {
		flex: 1 0 auto;
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		margin-top: 4rem;
		padding-bottom: 8.5rem;
		gap: 3rem;

		.card {
			width: 90%;
			max-width: 90rem;

			p {
				font-size: 1.4rem;
				letter-spacing: -0.03rem;
				color: rgba($secondary-color, 0.65);
				padding-bottom: 0.4rem;
			}

			.classification {
				align-items: center;
				display: flex;
				width: 100%;
				background-color: $tertiary-color;
				border-radius: 0.5rem;
				box-shadow:
					0rem 0.4rem 0.8rem 0.3rem rgba(#000, 0.15),
					0rem 0.1rem 0.3rem 0rem rgba(#000, 0.3);

				.ct-scan {
					width: 45%;
					height: 100%;
					object-fit: cover;
					border-radius: 0.5rem 0rem 0rem 0.5rem;
				}

				.prediction {
					padding: 1.6rem;
					font-size: 1.8rem;
					font-weight: bold;
					color: $secondary-color;
					width: 65%;

					.result {
						padding: 0.8rem 0rem 1.6rem 1rem;

						h3 {
							display: flex;
							gap: 0.3rem;
							align-items: center;
							vertical-align: center;
							font-size: 2rem;
							font-weight: normal;
							color: rgba($secondary-color, 0.55);

							&.selected {
								font-size: 2.2rem;
								font-weight: bold;
								color: rgba($secondary-color, 0.6);
							}

							p {
								margin: 0rem;
								padding: 0rem;
								font-size: 2rem;
								color: rgba($secondary-color, 0.55);

								&.selected {
									color: $primary-color;
									font-size: 2.2rem;
								}
							}
						}
					}

					.gradcam {
						display: flex;
						justify-content: center;
						align-items: center;
						width: 100%;
						height: 4.8rem;
						border: 0.35rem solid $primary-color;
						border-radius: 0.8rem;
						font-family: inherit;
						font-weight: normal;
						font-size: 1.8rem;
						background-color: transparent;
						color: $secondary-color;
						letter-spacing: 0.12rem;
						cursor: pointer;
					}

					.centered {
						text-align: center;
					}
				}
			}
		}
	}

	.navbar {
		position: sticky;
		bottom: 0rem;
		left: 0rem;
		width: 100%;
		height: 10.8rem;
		text-align: center;
		background-color: rgba($quaternary-color, 0.4);
		backdrop-filter: blur(1.6rem);
		border-top: 0.3rem solid rgba($quaternary-color, 0.3);
		display: grid;
		grid-template-columns: 5fr 1fr;
		grid-template-rows: 1fr;
		grid-column-gap: 1rem;
		grid-row-gap: 1rem;
		place-items: center;
		padding: 0rem 2.6rem;

		.file-select {
			width: 100%;

			.file-label {
				text-align: left;
				margin-bottom: 0.8rem;
				font-size: 1.4rem;
				color: $secondary-color;
			}

			.file-box {
				display: flex;
				height: 4.8rem;
				align-items: center;
				gap: 0.8rem;
				background-color: $tertiary-color;
				color: rgba($secondary-color, 0.7);
				border: 0.2rem solid rgba($secondary-color, 0.3);
				border-radius: 0.8rem;
				cursor: pointer;
				user-select: none;
				padding: 1.2rem;

				div {
					text-align: center;
					font-size: 1.6rem;
					white-space: nowrap;
					overflow: hidden;
					text-overflow: ellipsis;
				}
			}

			input[type='file'] {
				position: absolute;
				opacity: 0;
				width: 0;
				height: 0;
				pointer-events: none;
			}
		}

		.classify {
			width: 100%;
			padding-top: 2.56rem;

			.left-icon {
				display: none;
			}

			button {
				display: flex;
				justify-content: center;
				align-items: center;
				width: 100%;
				height: 4.8rem;
				border-radius: 0.8rem;
				border: none;
				font-family: inherit;
				font-weight: bold;
				font-size: 1.8rem;
				background-color: $primary-color;
				color: rgba($secondary-color, 0.7);
				letter-spacing: 0.12rem;
				cursor: pointer;
			}
		}
	}

	button {
		svg {
			transition: transform 0.3s cubic-bezier(0.76, 0, 0.24, 1);
		}

		&:hover svg {
			animation: bounce 0.6s infinite;
		}
	}

	@keyframes bounce {
		0%,
		100% {
			transform: translateY(0rem);
			animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
		}

		33% {
			transform: translateY(-0.35rem);
			animation-timing-function: cubic-bezier(0.76, 0, 0.24, 1);
		}
	}

	@media (max-width: $laptop) {
		.navbar {
			grid-template-columns: 3fr 1fr;
			padding: 0rem 2rem;
		}
	}

	@media (max-width: $tablet) {
		header {
			padding-top: 3.6rem;

			.title {
				font-size: 3.6rem;
			}

			.subtext {
				font-size: 1.8rem;
			}
		}

		.content {
			.card {
				.classification {
					h2 {
						font-size: 2rem;
					}

					.prediction {
						.result {
							h3 {
								font-size: 1.8rem;

								&.selected {
									font-size: 2rem;
								}

								p {
									font-size: 1.8rem;

									&.selected {
										font-size: 2rem;
									}
								}
							}
						}
					}
				}
			}
		}

		.navbar {
			grid-template-columns: 1.5fr 1fr;
			padding: 0rem 1.6rem;
		}
	}

	@media (max-width: $mobile) {
		header {
			padding: 3rem 1rem 0rem;

			.title {
				font-size: 3.2rem;
			}

			.subtext {
				font-size: 1.6rem;
			}
		}

		.content {
			.card {
				p {
					text-align: center;
				}

				.classification {
					flex-direction: column;
					border-radius: 0.5rem;

					h2 {
						font-size: 2rem;
					}

					.ct-scan {
						width: 100%;
						height: 45%;
						border-radius: 0.5rem 0.5rem 0rem 0rem;
					}

					.prediction {
						padding: 2.4rem;
						width: 100%;
						height: 65%;

						.result {
							h3 {
								font-size: 1.6rem;

								&.selected {
									font-size: 1.8rem;
								}

								p {
									font-size: 1.6rem;

									&.selected {
										font-size: 1.8rem;
									}
								}
							}
						}
					}
				}
			}
		}

		.navbar {
			height: 16.8rem;
			grid-template-columns: 1fr;
			grid-template-rows: 1fr 1fr;
			place-items: stretch;
			padding: 1.2rem 1rem 0rem;

			.file-select {
				align-self: end;
			}

			.classify {
				align-self: start;
				padding-top: 0rem;

				.left-icon {
					display: inherit;
				}
			}
		}
	}
</style>
